import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Talks to a model that was trained on dialogue text ("User: ..." and "Bot: ..." lines, see DialogueData).
 *
 *   java DialogueChat name=talk                      fixed test questions, then a chat
 *   java DialogueChat name=talk data=dialogue.txt chars=1500000
 *                                                    the same, and each probe reply is also marked [in training] when it appears
 *                                                    word for word as a bot reply in the training text (use the same data= and chars=
 *                                                    as the training run)
 *   Other options: temp=0.5 (chat temperature; lower = safer and plainer), probes=0 (skip the test questions).
 *
 * The model is not told what to say: it continues the text "User: <your line>\nBot:" and the reply is cut at the end of the line.
 */
public class DialogueChat {
    static final String[] PROBES = {
        "Hi", "Hello!", "How are you?", "What is your name?", "Thank you!", "Goodbye!",
        "Can you help me?", "What is the capital of France?"
    };
    static final double[] PROBE_TEMPS = { 0.3, 0.7 };
    static final int MAX_REPLY_TOKENS = 120;
    static final int HISTORY_CHARS = 1000;
    static final String START = "\n\n";                      // training conversations are separated by a blank line

    public static void main(String[] args) throws Exception {
        String name = null, data = null;
        int chars = 0;
        double temp = 0.5;
        boolean probes = true;
        List<String> known = Arrays.asList("name", "data", "chars", "temp", "probes");
        for (String a : args) {
            String[] kv = a.split("=", 2);
            if (kv.length != 2 || !known.contains(kv[0])) { System.out.println("Unknown argument '" + a + "'. Valid options: " + known); return; }
            switch (kv[0]) {
                case "name": name = kv[1]; break;
                case "data": data = kv[1]; break;
                case "chars": chars = Integer.parseInt(kv[1]); break;
                case "temp": temp = Double.parseDouble(kv[1]); break;
                default: probes = !kv[1].equals("0");
            }
        }
        String path = (name == null ? "" : name + "-") + "best.bin";
        MultiLayerTransformer model = MultiLayerTransformer.load(path);
        System.out.printf("Loaded %s: %,d parameters, sees %d tokens at once, %d tokens in the vocabulary.%n",
            path, model.parameterCount(), model.contextLength, model.tokenizer.size());
        if (!model.tokenizer.knows('\n') || !model.tokenizer.knows(':')) {
            System.out.println("This model has no newline or ':' tokens, so it was not trained on User:/Bot: text. Train it on dialogue.txt first.");
            return;
        }

        Set<String> trainReplies = null;
        if (data != null) {
            String text = ChatTransformer.clean(new String(Files.readAllBytes(Paths.get(data)), StandardCharsets.UTF_8));
            if (chars > 0) text = text.substring(0, Math.min(chars, text.length()));
            trainReplies = new HashSet<>();
            for (String line : text.split("\n")) if (line.startsWith("Bot: ")) trainReplies.add(line.substring(5).trim());
        }

        if (probes) runProbes(model, trainReplies);

        System.out.println("\nChat (temperature " + temp + "). The model only remembers the last " + model.contextLength + " tokens. Type 'exit' to quit, 'reset' to start a new conversation.\n");
        Random rand = new Random();
        Scanner scanner = new Scanner(System.in);
        String history = START;
        while (true) {
            System.out.print("You: ");
            if (!scanner.hasNextLine()) break;
            String input = scanner.nextLine();
            if (input.equalsIgnoreCase("exit")) { System.out.println("Goodbye!"); break; }
            if (input.equalsIgnoreCase("reset")) { history = START; System.out.println("(new conversation)\n"); continue; }
            String line = model.tokenizer.filter(input.replaceAll("\\s+", " ")).trim();
            if (line.length() < input.trim().length()) System.out.println("(Skipped " + (input.trim().length() - line.length()) + " character(s) the model has never seen.)");
            if (line.isEmpty()) continue;
            String prompt = history + "User: " + line + "\nBot:";
            String reply = reply(model, prompt, rand, temp);
            System.out.println("Bot: " + (reply.isEmpty() ? "(no reply)" : reply) + "\n");
            history = prompt + " " + reply + "\n";
            if (history.length() > HISTORY_CHARS) history = history.substring(history.length() - HISTORY_CHARS);
        }
        scanner.close();
    }

    /** Fresh conversation per question; two temperatures. A few samples are for reading, not for averaging. */
    static void runProbes(MultiLayerTransformer model, Set<String> trainReplies) {
        System.out.println("\nTest questions (each one starts a new conversation; one sample per temperature):");
        int total = 0, copied = 0, empty = 0;
        for (String q : PROBES) {
            System.out.println("You: " + q);
            for (double t : PROBE_TEMPS) {
                String line = model.tokenizer.filter(q);
                String reply = reply(model, START + "User: " + line + "\nBot:", new Random(7), t);
                total++;
                String mark = "";
                if (reply.isEmpty()) empty++;
                else if (trainReplies != null) { boolean in = trainReplies.contains(reply); if (in) copied++; mark = in ? "   [in training]" : "   [new]"; }
                System.out.printf("  t=%.1f  %s%s%n", t, reply.isEmpty() ? "(no reply)" : reply, mark);
            }
        }
        if (trainReplies != null)
            System.out.printf("%d of %d replies appear word for word as a bot reply in the training text. Short stock replies are expected to repeat; look at whether the answers fit the question.%n", copied, total);
        if (empty > 0) System.out.println(empty + " of " + total + " replies were empty.");
    }

    /** The model continues `prompt` (which ends with "Bot:") until the end of the line. */
    static String reply(MultiLayerTransformer m, String prompt, Random rand, double temp) {
        int[] ids = m.tokenizer.encode(prompt);
        int[] buf = Arrays.copyOf(ids, ids.length + MAX_REPLY_TOKENS);
        int len = ids.length;
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < MAX_REPLY_TOKENS; i++) {
            int from = Math.max(0, len - m.contextLength);
            int id = m.sampleNextId(Arrays.copyOfRange(buf, from, len), rand, temp);
            buf[len++] = id;
            out.append(m.tokenizer.tokenText(id));
            int nl = out.indexOf("\n");
            if (nl >= 0) { out.setLength(nl); break; }
        }
        return out.toString().trim();
    }
}
