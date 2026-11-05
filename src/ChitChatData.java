import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Step 25: builds a chit-chat training file ("User:" / "Bot:" lines) from a conversation dataset on
 * Hugging Face, in the same format DialogueChat and ChatTransformer already use.
 *
 * Default source: google/Synthetic-Persona-Chat (license CC-BY-4.0, so credit Google in the README; the
 * data file itself stays out of git). Its text column "Best Generated Conversation" holds lines starting
 * "User 1:" / "User 2:"; User 1 becomes "User" and User 2 becomes "Bot". Names are written into the data as
 * placeholders ("[user 1's name]"), so they are replaced: the user's name varies per conversation (the model
 * can learn to repeat the name it was given), the bot's name is always botname.
 * Other datasets with a list of utterances or role/content messages are also understood.
 *
 * Usage: java ChitChatData [out=chitchat.txt] [dataset=google/Synthetic-Persona-Chat] [config=default]
 *                          [train=train] [test=auto] [limit=4000] [testlimit=150] [botname=Sam]
 *                          [base=https://datasets-server.huggingface.co]
 * limit/testlimit = maximum conversations read from the training / held-out split (0 = all).
 * test=auto picks "validation" if the dataset has it, otherwise "test".
 */
public class ChitChatData {

    static final String SEPARATOR = "=====\n\n";
    static final String MARK = "\u0001".repeat(10);   // survives ChatTransformer.clean(), lets us find the train/held-out boundary
    static final int PAGE = 100;
    static final int WINDOW = 20;
    static final int VALCHARS = 20000;
    static final Set<String> KNOWN = new HashSet<>(Arrays.asList("out", "dataset", "config", "train", "test", "limit", "testlimit", "botname", "base"));

    public static void main(String[] args) throws Exception {
        Map<String, String> opt = new HashMap<>();
        opt.put("out", "chitchat.txt");
        opt.put("base", "https://datasets-server.huggingface.co");
        opt.put("dataset", "google/Synthetic-Persona-Chat");
        opt.put("config", "default");
        opt.put("train", "train");
        opt.put("test", "auto");
        opt.put("limit", "4000");
        opt.put("testlimit", "150");
        opt.put("botname", "Sam");
        for (String a : args) {
            int eq = a.indexOf('=');
            if (eq <= 0 || !KNOWN.contains(a.substring(0, eq))) {
                System.out.println("Unknown argument: " + a + "\nKnown: " + KNOWN);
                return;
            }
            opt.put(a.substring(0, eq), a.substring(eq + 1));
        }
        int limit = Integer.parseInt(opt.get("limit"));
        int testLimit = Integer.parseInt(opt.get("testlimit"));
        String testSplit = resolveTestSplit(opt);

        List<List<String[]>> train = fetchSplit(opt, opt.get("train"), limit);
        List<List<String[]>> test = fetchSplit(opt, testSplit, testLimit);
        if (train.isEmpty() || test.isEmpty()) {
            System.out.println("No usable conversations in one of the splits, so nothing was written.");
            return;
        }

        String full = render(train) + MARK + render(test);
        String cleaned = ChatTransformer.clean(full);
        int cut = cleaned.indexOf(MARK);
        if (cut < 0) throw new IllegalStateException("Boundary marker was removed by clean(); cannot split.");
        String trainText = cleaned.substring(0, cut);
        String testText = cleaned.substring(cut + MARK.length());
        String file = trainText + SEPARATOR + testText;
        Files.write(Paths.get(opt.get("out")), file.getBytes(StandardCharsets.UTF_8));

        int removed = full.length() - MARK.length() - trainText.length() - testText.length();
        System.out.printf("Wrote %s: %,d characters (%,d training + %,d held-out). Tidying removed %d rare characters.%n",
            opt.get("out"), file.length(), trainText.length(), testText.length(), removed);
        System.out.printf("Conversations: %,d training, %,d held-out.%n", train.size(), test.size());
        report(train, trainText, testText);

        int trainChars = trainText.length();
        int valStart = trainChars + SEPARATOR.length();
        System.out.println("\nTrain with (about 15 minutes):");
        System.out.println("java ChatTransformer data=" + opt.get("out") + " chars=" + trainChars + " valstart=" + valStart
            + " valchars=" + VALCHARS + " vocab=512 context=96 seconds=900 threads=8 name=chat");
        System.out.println("Then talk to it: java DialogueChat name=chat data=" + opt.get("out") + " chars=" + trainChars);
    }

    // ---------------------------------------------------------------- fetching

    /** Uses the test= option, or (auto) the first of "validation", "test" that the dataset really has. */
    static String resolveTestSplit(Map<String, String> opt) throws Exception {
        if (!opt.get("test").equals("auto")) return opt.get("test");
        Map<String, Object> reply = asMap(Json.parse(get(opt.get("base") + "/splits?dataset=" + enc(opt.get("dataset")))));
        Set<String> have = new LinkedHashSet<>();
        if (reply != null && reply.get("splits") instanceof List) {
            for (Object o : (List<?>) reply.get("splits")) {
                Map<String, Object> m = asMap(o);
                if (m != null && opt.get("config").equals(m.get("config")) && m.get("split") instanceof String) have.add((String) m.get("split"));
            }
        }
        for (String candidate : new String[] { "validation", "test" }) {
            if (have.contains(candidate)) {
                System.out.println("Splits of " + opt.get("dataset") + " (" + opt.get("config") + "): " + have + "; held-out text comes from '" + candidate + "'.");
                return candidate;
            }
        }
        throw new IllegalStateException("No 'validation' or 'test' split found. Splits seen: " + have + ". Run again with test=<name>.");
    }

    static List<List<String[]>> fetchSplit(Map<String, String> opt, String split, int limit) throws Exception {
        List<List<String[]>> out = new ArrayList<>();
        int offset = 0, total = Integer.MAX_VALUE, truncated = 0, tooShort = 0, unknownRoles = 0, leftoverBrackets = 0;
        String keyUsed = null;
        while (offset < total && (limit == 0 || out.size() < limit)) {
            String url = opt.get("base") + "/rows?dataset=" + enc(opt.get("dataset")) + "&config=" + enc(opt.get("config"))
                + "&split=" + enc(split) + "&offset=" + offset + "&length=" + PAGE;
            Object parsed = Json.parse(get(url));
            Map<String, Object> page = asMap(parsed);
            if (page == null || !(page.get("rows") instanceof List)) throw new IllegalStateException("Unexpected reply: " + url);
            total = ((Number) page.get("num_rows_total")).intValue();
            List<?> rows = (List<?>) page.get("rows");
            if (rows.isEmpty()) break;
            for (Object r : rows) {
                Map<String, Object> wrapper = asMap(r);
                Map<String, Object> row = wrapper == null ? null : asMap(wrapper.get("row"));
                if (row == null) continue;
                Object tc = wrapper.get("truncated_cells");
                if (tc instanceof List && !((List<?>) tc).isEmpty()) { truncated++; continue; }
                String[] key = new String[1];
                int[] unk = new int[1];
                Object idxObj = wrapper.get("row_idx");
                int rowIdx = idxObj instanceof Number ? ((Number) idxObj).intValue() : offset;
                List<String[]> turns = turnsOf(row, key, unk, rowIdx, opt.get("botname"));
                unknownRoles += unk[0];
                if (turns == null) {
                    throw new IllegalStateException("Cannot find the conversation in a row. Columns seen: " + row.keySet()
                        + "\nTell me this list and I will adapt the converter.");
                }
                keyUsed = key[0];
                turns = tidyTurns(turns);
                if (turns.size() < 2) { tooShort++; continue; }
                if (hasBracket(turns)) { leftoverBrackets++; continue; }
                out.add(turns);
                if (limit != 0 && out.size() >= limit) break;
            }
            offset += rows.size();
            System.out.printf("  %s: %,d of %,d rows read%n", split, Math.min(offset, total), total);
        }
        System.out.printf("%s: %,d conversations kept (column '%s'); skipped %d too short, %d with cut-off cells, %d with a leftover [placeholder], %d turns with an unknown speaker.%n",
            split, out.size(), keyUsed, tooShort, truncated, leftoverBrackets, unknownRoles);
        return out;
    }

    static String enc(String s) throws UnsupportedEncodingException {
        return URLEncoder.encode(s, "UTF-8").replace("%2F", "/");
    }

    static String get(String url) throws IOException, InterruptedException {
        String lastProblem = "";
        for (int attempt = 1; attempt <= 4; attempt++) {
            try {
                HttpURLConnection c = (HttpURLConnection) URI.create(url).toURL().openConnection();
                c.setRequestProperty("User-Agent", "mini-llm-java");
                c.setConnectTimeout(20000);
                c.setReadTimeout(60000);
                int code = c.getResponseCode();
                if (code == 200) return new String(c.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
                String body = c.getErrorStream() == null ? "" : new String(c.getErrorStream().readAllBytes(), StandardCharsets.UTF_8);
                lastProblem = "HTTP " + code + " " + (body.length() > 300 ? body.substring(0, 300) : body);
                if (code != 429 && code < 500) throw new IllegalStateException(lastProblem + "\n  for " + url);
            } catch (SocketTimeoutException | ConnectException e) {
                lastProblem = e.toString();
            }
            Thread.sleep(1500L * attempt);
        }
        throw new IOException("Giving up after 4 tries: " + lastProblem + "\n  for " + url);
    }

    // ---------------------------------------------------------------- turning rows into turns

    /** Returns {speaker ("User"/"Bot"), text} pairs, or null if the row has no recognizable conversation column. */
    static List<String[]> turnsOf(Map<String, Object> row, String[] keyUsed, int[] unknown, int rowIdx, String botName) {
        Object text = row.get("Best Generated Conversation");
        if (text instanceof String) {
            keyUsed[0] = "Best Generated Conversation";
            return personaTurns((String) text, rowIdx, botName, unknown);
        }
        for (String key : new String[] { "conversations", "messages", "dialog", "dialogue" }) {
            Object v = row.get(key);
            if (!(v instanceof List)) continue;
            keyUsed[0] = key;
            List<String[]> turns = new ArrayList<>();
            int idx = 0;
            for (Object item : (List<?>) v) {
                if (item instanceof String) {                       // plain list of utterances: speakers alternate
                    turns.add(new String[] { idx % 2 == 0 ? "User" : "Bot", (String) item });
                } else if (item instanceof Map) {
                    Map<String, Object> m = asMap(item);
                    String role = firstString(m, "from", "role", "speaker");
                    String said = firstString(m, "value", "content", "text");
                    if (role == null || said == null) continue;
                    role = role.toLowerCase();
                    if (role.equals("system")) continue;
                    if (role.equals("human") || role.equals("user")) turns.add(new String[] { "User", said });
                    else if (role.equals("gpt") || role.equals("assistant") || role.equals("bot")) turns.add(new String[] { "Bot", said });
                    else unknown[0]++;
                }
                idx++;
            }
            return turns;
        }
        return null;
    }


    static final String[] NAMES = { "Alex", "Maria", "Tom", "Emma", "Jack", "Sara", "Ben", "Lily", "Mark", "Anna",
                                    "Chris", "Nina", "Paul", "Julia", "Dan", "Kate" };

    /** Splits "User 1: ...\nUser 2: ..." text into turns (User 1 = "User", User 2 = "Bot") and fills in the name placeholders. */
    static List<String[]> personaTurns(String text, int rowIdx, String botName, int[] unknown) {
        String userName = NAMES[Math.floorMod(rowIdx * 7 + 3, NAMES.length)];
        if (userName.equalsIgnoreCase(botName)) userName = NAMES[Math.floorMod(rowIdx * 7 + 4, NAMES.length)];
        text = text.replaceAll("(?i)\\[\\s*user\\s*1(?:'s)?\\s*name\\s*\\]", userName)
                   .replaceAll("(?i)\\[\\s*user\\s*2(?:'s)?\\s*name\\s*\\]", java.util.regex.Matcher.quoteReplacement(botName));
        java.util.regex.Pattern line = java.util.regex.Pattern.compile("^\\s*User\\s*([12])\\s*:\\s*(.*)$");
        List<String[]> turns = new ArrayList<>();
        for (String l : text.split("\\r?\\n")) {
            if (l.trim().isEmpty()) continue;
            java.util.regex.Matcher m = line.matcher(l);
            if (m.matches()) {
                turns.add(new String[] { m.group(1).equals("1") ? "User" : "Bot", m.group(2) });
            } else if (!turns.isEmpty()) {              // a line without a speaker label continues the previous turn
                turns.get(turns.size() - 1)[1] += " " + l.trim();
                unknown[0]++;
            }
        }
        return turns;
    }

    static boolean hasBracket(List<String[]> turns) {
        for (String[] t : turns) if (t[1].matches(".*\\[[^\\]]*\\].*")) return true;
        return false;
    }

    static String firstString(Map<String, Object> m, String... keys) {
        for (String k : keys) if (m.get(k) instanceof String) return (String) m.get(k);
        return null;
    }

    /** Cleans each turn, merges consecutive turns of the same speaker, drops a leading Bot turn and a trailing unanswered User turn. */
    static List<String[]> tidyTurns(List<String[]> raw) {
        List<String[]> out = new ArrayList<>();
        for (String[] t : raw) {
            String text = tidy(t[1]);
            if (text.isEmpty()) continue;
            if (out.isEmpty() && t[0].equals("Bot")) continue;
            if (!out.isEmpty() && out.get(out.size() - 1)[0].equals(t[0])) {
                String[] last = out.get(out.size() - 1);
                last[1] = last[1] + " " + text;
            } else {
                out.add(new String[] { t[0], text });
            }
        }
        if (!out.isEmpty() && out.get(out.size() - 1)[0].equals("User")) out.remove(out.size() - 1);
        return out;
    }

    /** Collapses whitespace and repairs word-split punctuation ("How are you ?" becomes "How are you?", "do n't" becomes "don't"). */
    static String tidy(String t) {
        t = t.replace('‘', '\'').replace('’', '\'').replace('“', '"').replace('”', '"');
        t = t.replaceAll("\\s+", " ").trim();
        t = t.replaceAll(" ([.,!?;:])", "$1");
        t = t.replaceAll("(\\w) ' (\\w)", "$1'$2");
        t = t.replaceAll("(\\w) (n't|'s|'m|'re|'ve|'ll|'d)\\b", "$1$2");
        return t;
    }

    static String render(List<List<String[]>> convs) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < convs.size(); i++) {
            if (i > 0) sb.append(SEPARATOR);
            for (String[] t : convs.get(i)) sb.append(t[0]).append(": ").append(t[1]).append('\n');
        }
        return sb.toString();
    }

    // ---------------------------------------------------------------- report

    static void report(List<List<String[]>> train, String trainText, String testText) {
        Map<String, Integer> botCounts = new HashMap<>(), firstBot = new HashMap<>();
        int botLines = 0, userLines = 0, spaced = 0;
        for (List<String[]> conv : train) {
            boolean first = true;
            for (String[] t : conv) {
                if (t[0].equals("User")) { userLines++; continue; }
                botLines++;
                botCounts.merge(t[1], 1, Integer::sum);
                if (first) { firstBot.merge(t[1], 1, Integer::sum); first = false; }
                if (t[1].contains(" ?") || t[1].contains(" .") || t[1].contains(" !")) spaced++;
            }
        }
        int repeated = 0;
        for (int c : botCounts.values()) if (c > 1) repeated += c;
        System.out.printf("Training turns: %,d user lines, %,d bot lines (%.1f per conversation); %,d different bot replies; %.1f%% of bot lines are a reply that appears more than once.%n",
            userLines, botLines, (double) (userLines + botLines) / train.size(), botCounts.size(), 100.0 * repeated / Math.max(1, botLines));
        System.out.printf("First bot reply of each conversation: %,d different ones in %,d conversations (a fixed greeting would show 1-2 here).%n",
            firstBot.size(), train.size());
        System.out.println("Most common bot replies:");
        printTop(botCounts, 6);
        System.out.println("Most common first bot replies:");
        printTop(firstBot, 4);
        System.out.printf("Bot lines still showing split punctuation (' ?', ' .'): %,d (%.2f%%).%n", spaced, 100.0 * spaced / Math.max(1, botLines));

        Map<String, Integer> afterHow = new HashMap<>();
        int howCount = 0;
        for (List<String[]> conv : train) {
            for (int i = 0; i + 1 < conv.size(); i++) {
                if (conv.get(i)[0].equals("User") && conv.get(i)[1].toLowerCase().contains("how are you")) {
                    howCount++;
                    afterHow.merge(conv.get(i + 1)[1], 1, Integer::sum);
                }
            }
        }
        System.out.printf("User lines containing \"how are you\": %,d (the earlier dialogue.txt had 0). The bot's replies to them: %,d different.%n", howCount, afterHow.size());
        printTop(afterHow, 5);

        String held = testText.length() > VALCHARS ? testText.substring(0, VALCHARS) : testText;
        long[] trainHashes = windowHashes(trainText);
        Arrays.sort(trainHashes);
        long[] heldHashes = windowHashes(held);
        int hits = 0;
        for (long h : heldHashes) if (Arrays.binarySearch(trainHashes, h) >= 0) hits++;
        System.out.printf("Held-out dialogue (first %,d chars, the part used for 'val'): %.1f%% of its %d-character pieces appear word for word in the training text (the baseline for 'copied').%n",
            held.length(), 100.0 * hits / Math.max(1, heldHashes.length), WINDOW);

        System.out.println("\nFirst training conversation:");
        int shown = 0;
        for (String[] t : train.get(0)) {
            if (shown++ >= 10) { System.out.println("  ..."); break; }
            System.out.println("  " + t[0] + ": " + t[1]);
        }
    }

    static void printTop(Map<String, Integer> counts, int n) {
        List<Map.Entry<String, Integer>> es = new ArrayList<>(counts.entrySet());
        es.sort((a, b) -> b.getValue() - a.getValue());
        for (int i = 0; i < Math.min(n, es.size()); i++) {
            if (es.get(i).getValue() < 2 && i > 0) break;
            String s = es.get(i).getKey();
            System.out.printf("  %5d x  %s%n", es.get(i).getValue(), s.length() > 90 ? s.substring(0, 90) + "..." : s);
        }
    }

    /** 64-bit rolling hash of every WINDOW-character piece of s. */
    static long[] windowHashes(String s) {
        int n = s.length() - WINDOW + 1;
        if (n <= 0) return new long[0];
        final long B = 1000003L;
        long pow = 1;
        for (int i = 0; i < WINDOW - 1; i++) pow *= B;
        long h = 0;
        for (int i = 0; i < WINDOW; i++) h = h * B + s.charAt(i);
        long[] out = new long[n];
        out[0] = h;
        for (int i = 1; i < n; i++) {
            h = (h - s.charAt(i - 1) * pow) * B + s.charAt(i + WINDOW - 1);
            out[i] = h;
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> asMap(Object o) {
        return o instanceof Map ? (Map<String, Object>) o : null;
    }

    // ---------------------------------------------------------------- minimal JSON reader (no libraries)

    static final class Json {
        final String s;
        int i = 0;

        Json(String s) { this.s = s; }

        static Object parse(String s) {
            Json j = new Json(s);
            j.ws();
            Object v = j.value();
            j.ws();
            if (j.i != s.length()) throw new IllegalArgumentException("Extra data after JSON at position " + j.i);
            return v;
        }

        void ws() { while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++; }

        Object value() {
            if (i >= s.length()) throw new IllegalArgumentException("Unexpected end of JSON");
            char c = s.charAt(i);
            if (c == '{') return object();
            if (c == '[') return array();
            if (c == '"') return string();
            if (s.startsWith("true", i)) { i += 4; return Boolean.TRUE; }
            if (s.startsWith("false", i)) { i += 5; return Boolean.FALSE; }
            if (s.startsWith("null", i)) { i += 4; return null; }
            return number();
        }

        Map<String, Object> object() {
            Map<String, Object> m = new LinkedHashMap<>();
            i++;
            ws();
            if (s.charAt(i) == '}') { i++; return m; }
            while (true) {
                ws();
                String k = string();
                ws();
                if (s.charAt(i) != ':') throw new IllegalArgumentException("Expected ':' at " + i);
                i++;
                ws();
                m.put(k, value());
                ws();
                char c = s.charAt(i++);
                if (c == '}') return m;
                if (c != ',') throw new IllegalArgumentException("Expected ',' or '}' at " + (i - 1));
            }
        }

        List<Object> array() {
            List<Object> l = new ArrayList<>();
            i++;
            ws();
            if (s.charAt(i) == ']') { i++; return l; }
            while (true) {
                ws();
                l.add(value());
                ws();
                char c = s.charAt(i++);
                if (c == ']') return l;
                if (c != ',') throw new IllegalArgumentException("Expected ',' or ']' at " + (i - 1));
            }
        }

        String string() {
            if (s.charAt(i) != '"') throw new IllegalArgumentException("Expected string at " + i);
            i++;
            StringBuilder sb = new StringBuilder();
            while (true) {
                char c = s.charAt(i++);
                if (c == '"') return sb.toString();
                if (c != '\\') { sb.append(c); continue; }
                char e = s.charAt(i++);
                switch (e) {
                    case 'n': sb.append('\n'); break;
                    case 't': sb.append('\t'); break;
                    case 'r': sb.append('\r'); break;
                    case 'b': sb.append('\b'); break;
                    case 'f': sb.append('\f'); break;
                    case 'u': sb.append((char) Integer.parseInt(s.substring(i, i + 4), 16)); i += 4; break;
                    default: sb.append(e);   // \" \\ \/
                }
            }
        }

        Number number() {
            int start = i;
            while (i < s.length() && "+-0123456789.eE".indexOf(s.charAt(i)) >= 0) i++;
            if (start == i) throw new IllegalArgumentException("Unexpected character '" + s.charAt(i) + "' at " + i);
            return Double.parseDouble(s.substring(start, i));
        }
    }
}
