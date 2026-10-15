import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.*;

/**
 * Turns text into token numbers and back.
 *
 * Every distinct character is a token (ids 0..C-1, in sorted order). On top of that, byte-pair encoding (BPE) learns
 * "merges": it finds the pair of neighbouring tokens that occurs most often in the training text, gives that pair a new
 * token id, and repeats. Frequent words and word-parts such as " the", "ing" or "Lily" therefore become single tokens, so a
 * window of 32 tokens covers about 100 characters instead of 32, and the model no longer has to spell every letter.
 * With zero merges this is plain character-level tokenization, exactly what the model used before.
 *
 * Merges never cross a word boundary: the text is first cut into pieces (a word with its leading space, a number, a run of
 * punctuation, or a run of whitespace) and tokens are merged only inside a piece.
 */
public class Tokenizer {
    private static final Pattern PIECE = Pattern.compile(" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\\s]+|\\s+");

    final char[] baseChars;
    final int[] mergeA, mergeB;          // merge k builds token (C + k) out of tokens mergeA[k] and mergeB[k]
    final int size;                      // total number of tokens
    long trainTokens;                    // how many tokens the training text became (only set by train(), used by the checks)
    private final int C;
    private final String[] text;         // the characters each token stands for
    private final Map<Character, Integer> charId = new HashMap<>();
    private final Map<Long, Integer> mergedId = new HashMap<>();
    private final Map<String, int[]> cache = new ConcurrentHashMap<>();

    public Tokenizer(char[] baseChars, int[] mergeA, int[] mergeB) {
        this.baseChars = baseChars;
        this.mergeA = mergeA;
        this.mergeB = mergeB;
        C = baseChars.length;
        size = C + mergeA.length;
        text = new String[size];
        for (int i = 0; i < C; i++) { text[i] = String.valueOf(baseChars[i]); charId.put(baseChars[i], i); }
        for (int k = 0; k < mergeA.length; k++) {
            text[C + k] = text[mergeA[k]] + text[mergeB[k]];
            mergedId.put(key(mergeA[k], mergeB[k]), C + k);
        }
    }

    private static long key(int a, int b) { return ((long) a << 32) | (b & 0xffffffffL); }

    /** One token per distinct character of `text` (sorted), no merges. */
    public static Tokenizer chars(String text) {
        Set<Character> unique = new TreeSet<>();
        for (int i = 0; i < text.length(); i++) unique.add(text.charAt(i));
        char[] base = new char[unique.size()];
        int i = 0;
        for (char c : unique) base[i++] = c;
        return new Tokenizer(base, new int[0], new int[0]);
    }

    /** Learns merges from `text` until there are `vocabSize` tokens in total (or no pair occurs twice any more). */
    public static Tokenizer train(String text, int vocabSize) {
        Tokenizer base = chars(text);
        int C = base.C;

        // Count each distinct piece once; the merges are then learned from these counts, which is much faster than rescanning the text.
        Map<String, Integer> counts = new HashMap<>();
        Matcher m = PIECE.matcher(text);
        while (m.find()) counts.merge(m.group(), 1, Integer::sum);
        List<String> pieces = new ArrayList<>(counts.keySet());
        Collections.sort(pieces);                                   // fixed order, so the result never depends on hash ordering
        int n = pieces.size();
        int[][] seq = new int[n][];
        long[] freq = new long[n];
        for (int i = 0; i < n; i++) {
            String p = pieces.get(i);
            seq[i] = new int[p.length()];
            for (int j = 0; j < p.length(); j++) seq[i][j] = base.charId.get(p.charAt(j));
            freq[i] = counts.get(p);
        }

        List<int[]> merges = new ArrayList<>();
        for (int nextId = C; nextId < vocabSize; nextId++) {
            Map<Long, Long> pairCount = new HashMap<>();
            for (int i = 0; i < n; i++) {
                int[] s = seq[i];
                for (int j = 0; j + 1 < s.length; j++) pairCount.merge(key(s[j], s[j + 1]), freq[i], Long::sum);
            }
            long bestKey = -1, bestCount = 1;                       // a pair must occur at least twice to be worth a token
            for (Map.Entry<Long, Long> e : pairCount.entrySet()) {
                long c = e.getValue(), k = e.getKey();
                if (c > bestCount || (c == bestCount && bestKey != -1 && k < bestKey)) { bestCount = c; bestKey = k; }
            }
            if (bestKey == -1) break;
            int a = (int) (bestKey >>> 32), b = (int) (bestKey & 0xffffffffL);
            merges.add(new int[] { a, b });
            for (int i = 0; i < n; i++) seq[i] = applyMerge(seq[i], a, b, nextId);
        }

        int[] ma = new int[merges.size()], mb = new int[merges.size()];
        for (int k = 0; k < ma.length; k++) { ma[k] = merges.get(k)[0]; mb[k] = merges.get(k)[1]; }
        Tokenizer t = new Tokenizer(base.baseChars, ma, mb);
        long total = 0;
        for (int i = 0; i < n; i++) total += freq[i] * seq[i].length;
        t.trainTokens = total;
        return t;
    }

    private static int[] applyMerge(int[] s, int a, int b, int id) {
        boolean found = false;
        for (int j = 0; j + 1 < s.length && !found; j++) found = s[j] == a && s[j + 1] == b;
        if (!found) return s;
        int[] out = new int[s.length];
        int w = 0;
        for (int j = 0; j < s.length; ) {
            if (j + 1 < s.length && s[j] == a && s[j + 1] == b) { out[w++] = id; j += 2; }
            else out[w++] = s[j++];
        }
        return Arrays.copyOf(out, w);
    }

    public int size() { return size; }
    public int numMerges() { return mergeA.length; }
    public String tokenText(int id) { return text[id]; }
    public boolean knows(char c) { return charId.containsKey(c); }

    /** `s` without the characters that have no token (they cannot be represented, so they are skipped). */
    public String filter(String s) {
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) if (charId.containsKey(s.charAt(i))) sb.append(s.charAt(i));
        return sb.toString();
    }

    /** Token ids for `s`; characters without a token are skipped. */
    public int[] encode(String s) {
        String t = filter(s);
        if (mergedId.isEmpty()) {
            int[] ids = new int[t.length()];
            for (int i = 0; i < ids.length; i++) ids[i] = charId.get(t.charAt(i));
            return ids;
        }
        int[] buf = new int[t.length()];
        int len = 0;
        Matcher m = PIECE.matcher(t);
        while (m.find()) {
            int[] piece = encodePiece(m.group());
            System.arraycopy(piece, 0, buf, len, piece.length);
            len += piece.length;
        }
        return Arrays.copyOf(buf, len);
    }

    /** Applies the learned merges to one piece, always the earliest-learned merge first (the order they were learned in). */
    private int[] encodePiece(String piece) {
        int[] cached = cache.get(piece);
        if (cached != null) return cached;
        int[] ids = new int[piece.length()];
        for (int i = 0; i < ids.length; i++) ids[i] = charId.get(piece.charAt(i));
        int len = ids.length;
        while (len > 1) {
            int bestId = Integer.MAX_VALUE;
            for (int i = 0; i + 1 < len; i++) {
                Integer id = mergedId.get(key(ids[i], ids[i + 1]));
                if (id != null && id < bestId) bestId = id;
            }
            if (bestId == Integer.MAX_VALUE) break;
            int a = mergeA[bestId - C], b = mergeB[bestId - C];
            int w = 0;
            for (int i = 0; i < len; ) {
                if (i + 1 < len && ids[i] == a && ids[i + 1] == b) { ids[w++] = bestId; i += 2; }
                else ids[w++] = ids[i++];
            }
            len = w;
        }
        int[] out = Arrays.copyOf(ids, len);
        cache.put(piece, out);
        return out;
    }

    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) sb.append(text[id]);
        return sb.toString();
    }

    void write(DataOutputStream out) throws IOException {
        out.writeUTF(new String(baseChars));
        out.writeInt(mergeA.length);
        for (int k = 0; k < mergeA.length; k++) { out.writeInt(mergeA[k]); out.writeInt(mergeB[k]); }
    }

    /** Reads what write() wrote. */
    static Tokenizer read(DataInputStream in) throws IOException {
        char[] base = in.readUTF().toCharArray();
        int n = in.readInt();
        int[] a = new int[n], b = new int[n];
        for (int k = 0; k < n; k++) { a[k] = in.readInt(); b[k] = in.readInt(); }
        return new Tokenizer(base, a, b);
    }
}
