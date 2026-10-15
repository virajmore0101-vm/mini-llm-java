import java.io.*;
import java.nio.file.*;
import java.util.*;

/** Checks the tokenizer: decoding must give back exactly what was encoded, training must be repeatable, and saving must not change anything. */
public class TokenizerCheck {
    static int failures = 0;

    static void check(String what, boolean ok) {
        System.out.printf("  %-70s %s%n", what, ok ? "(OK)" : "(FAIL)");
        if (!ok) failures++;
    }

    public static void main(String[] args) throws Exception {
        String all = Files.readString(Paths.get("shakespeare.txt"));
        String train = all.substring(0, 200000), other = all.substring(500000, 520000);
        int vocab = 400;

        long t0 = System.currentTimeMillis();
        Tokenizer tok = Tokenizer.train(train, vocab);
        double seconds = (System.currentTimeMillis() - t0) / 1000.0;
        int[] ids = tok.encode(train);
        System.out.printf("TOKENIZER: trained %d tokens (%d characters + %d merges) in %.1fs; training text %,d characters -> %,d tokens (%.2f characters per token)%n",
            tok.size(), tok.size() - tok.numMerges(), tok.numMerges(), seconds, train.length(), ids.length, (double) train.length() / ids.length);

        check("decode(encode(text)) == text on the training text", tok.decode(ids).equals(train));
        check("decode(encode(text)) == text on text it was not trained on", tok.decode(tok.encode(other)).equals(other));
        check("encoding the training text gives as many tokens as training counted", ids.length == tok.trainTokens);
        check("vocabulary is full (no merge was skipped)", tok.size() == vocab);

        // odd input: unknown characters are skipped, empty text, whitespace runs, punctuation clusters, repeated letters
        String odd = "  Hello,\n\n  world!!!  aaaa  \té中 123 ... ";
        check("unknown characters are skipped, the rest round-trips", tok.decode(tok.encode(odd)).equals(tok.filter(odd)));
        check("empty text gives no tokens", tok.encode("").length == 0 && tok.decode(new int[0]).isEmpty());

        // a second training run must give identical merges
        Tokenizer again = Tokenizer.train(train, vocab);
        check("training twice gives identical merges", Arrays.equals(tok.mergeA, again.mergeA) && Arrays.equals(tok.mergeB, again.mergeB));

        // save and load
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        tok.write(new DataOutputStream(bytes));
        Tokenizer loaded = Tokenizer.read(new DataInputStream(new ByteArrayInputStream(bytes.toByteArray())));
        check("a saved and reloaded tokenizer encodes exactly the same", Arrays.equals(loaded.encode(other), tok.encode(other)) && loaded.size() == tok.size());

        // zero merges is plain characters
        Tokenizer plain = Tokenizer.chars(train);
        int[] plainIds = plain.encode(other);
        boolean oneEach = true;
        String known = plain.filter(other);
        for (int i = 0; i < plainIds.length; i++) oneEach &= plain.tokenText(plainIds[i]).charAt(0) == known.charAt(i);
        check("without merges every character is its own token", plainIds.length == known.length() && oneEach);

        // every token's text is what decoding it alone gives, and no token is empty
        boolean texts = true;
        for (int id = 0; id < tok.size(); id++) texts &= tok.tokenText(id).length() > 0 && tok.decode(new int[] { id }).equals(tok.tokenText(id));
        check("every token stands for at least one character", texts);

        // show a few of the learned tokens and one encoded sentence
        StringBuilder sb = new StringBuilder();
        for (int k = tok.size() - tok.numMerges(); k < tok.size() - tok.numMerges() + 12; k++) sb.append('[').append(tok.tokenText(k)).append("] ");
        System.out.println("  first merges: " + sb);
        StringBuilder last = new StringBuilder();
        for (int k = tok.size() - 12; k < tok.size(); k++) last.append('[').append(tok.tokenText(k)).append("] ");
        System.out.println("  last merges:  " + last);
        StringBuilder enc = new StringBuilder();
        for (int id : tok.encode("To be, or not to be, that is the question:")) enc.append('[').append(tok.tokenText(id)).append(']');
        System.out.println("  \"To be, or not to be, that is the question:\" -> " + enc);

        System.out.println(failures == 0 ? "TOKENIZER CHECKS: all OK" : "TOKENIZER CHECKS: " + failures + " FAILED");
    }
}
