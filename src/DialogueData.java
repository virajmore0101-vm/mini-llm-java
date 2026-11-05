import java.io.*;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Builds a dialogue training file ("User: ..." / "Bot: ..." lines, one blank line between conversations) from the
 * Hugging Face dataset HuggingFaceTB/everyday-conversations-llama3.1-2k (about 2,200 short generated conversations, Apache-2.0).
 * It downloads the rows through the dataset viewer's JSON service, so it needs no extra software, only Java 11 or newer.
 *
 *   java DialogueData                       download both splits and write dialogue.txt
 *   java DialogueData out=talk.txt          write somewhere else
 *   java DialogueData trainfile=a.jsonl testfile=b.jsonl     use your own files instead (one JSON object per line, each holding
 *                                                           a list of {"role": ..., "content": ...} messages)
 * Other options: base= dataset= config= train= test=  (service address, dataset name, config name, split names).
 *
 * The test split goes to the end of the file, after a ===== line, so it can be used as the held-out passage:
 * the program prints the exact ChatTransformer command (chars= and valstart= are filled in for you).
 * It writes nothing except the output file.
 */
public class DialogueData {
    static final String SEPARATOR = "=====\n\n";
    static final int PAGE = 100;

    public static void main(String[] args) throws Exception {
        Map<String, String> opt = new HashMap<>();
        List<String> known = Arrays.asList("out", "base", "dataset", "config", "train", "test", "trainfile", "testfile");
        for (String a : args) {
            String[] kv = a.split("=", 2);
            if (kv.length != 2 || !known.contains(kv[0])) { System.out.println("Unknown argument '" + a + "'. Valid options: " + known); return; }
            opt.put(kv[0], kv[1]);
        }
        String out = opt.getOrDefault("out", "dialogue.txt");
        String base = opt.getOrDefault("base", "https://datasets-server.huggingface.co");
        String dataset = opt.getOrDefault("dataset", "HuggingFaceTB/everyday-conversations-llama3.1-2k");
        String config = opt.getOrDefault("config", "default");

        List<Object> trainRows, testRows;
        if (opt.containsKey("trainfile") && opt.containsKey("testfile")) {
            trainRows = readLines(opt.get("trainfile"));
            testRows = readLines(opt.get("testfile"));
        } else {
            HttpClient client = HttpClient.newHttpClient();
            trainRows = fetchSplit(client, base, dataset, config, opt.getOrDefault("train", "train_sft"));
            testRows = fetchSplit(client, base, dataset, config, opt.getOrDefault("test", "test_sft"));
        }

        List<String> train = conversations(trainRows, "train"), test = conversations(testRows, "test");
        if (train.isEmpty() || test.isEmpty()) { System.out.println("No conversations found, nothing written."); return; }

        String trainText = String.join("\n", train) + "\n", testText = String.join("\n", test);
        String whole = trainText + SEPARATOR + testText;
        if (whole.indexOf("=====") != whole.lastIndexOf("=====")) { System.out.println("The data itself contains '=====', so the split point would be ambiguous. Nothing written."); return; }

        // The training program tidies text the same way (curly quotes, rare characters), so tidy it here first and measure the split point afterwards.
        String cleaned = ChatTransformer.clean(whole);
        int cut = cleaned.indexOf("=====");
        if (cut < 0 || cleaned.indexOf("=====", cut + 1) >= 0 || !cleaned.startsWith(SEPARATOR, cut)) { System.out.println("Could not locate the split point after tidying. Nothing written."); return; }
        int valStart = cut + SEPARATOR.length();
        String trainClean = cleaned.substring(0, cut), testClean = cleaned.substring(valStart);
        if (!ChatTransformer.clean(cleaned).equals(cleaned)) { System.out.println("Tidying the output again changed it, so character positions would shift. Nothing written."); return; }
        Files.write(Paths.get(out), cleaned.getBytes(StandardCharsets.UTF_8));

        // ---- what is in the file ----
        int dropped = whole.length() - cleaned.length();
        System.out.printf("Wrote %s: %,d characters (%,d training + %,d held-out). Tidying removed %,d rare characters.%n", out, cleaned.length(), cut, testClean.length(), dropped);
        System.out.printf("Conversations: %,d training, %,d held-out.%n", train.size(), test.size());
        Map<String, Integer> replyCounts = new HashMap<>();
        int botLines = 0, userLines = 0;
        for (String line : trainClean.split("\n")) {
            if (line.startsWith("Bot: ")) { botLines++; replyCounts.merge(line.substring(5), 1, Integer::sum); }
            else if (line.startsWith("User: ")) userLines++;
        }
        int repeated = 0;
        for (int c : replyCounts.values()) if (c > 1) repeated += c;
        System.out.printf("Training turns: %,d user lines, %,d bot lines (%.1f per conversation); %,d different bot replies; %.1f%% of bot lines are a reply that appears more than once.%n",
            userLines, botLines, (userLines + botLines) / (double) train.size(), replyCounts.size(), botLines == 0 ? 0 : 100.0 * repeated / botLines);
        List<Map.Entry<String, Integer>> top = new ArrayList<>(replyCounts.entrySet());
        top.sort((a, b) -> b.getValue() - a.getValue());
        System.out.println("Most common bot replies:");
        for (int i = 0; i < Math.min(5, top.size()); i++) System.out.printf("  %4d x  %s%n", top.get(i).getValue(), top.get(i).getKey());

        // Baseline for memorization checks: how much of real held-out dialogue already appears word for word in the training text.
        String probe = testClean.substring(0, Math.min(30000, testClean.length()));
        int found = 0, checked = 0;
        for (int i = 0; i + 20 <= probe.length(); i += 5) { checked++; if (trainClean.contains(probe.substring(i, i + 20))) found++; }
        System.out.printf("Held-out dialogue: %.1f%% of its 20-character pieces appear word for word in the training text (the baseline for 'copied').%n", checked == 0 ? 0 : 100.0 * found / checked);

        System.out.println("\nFirst training conversation:");
        int shown = 0;
        for (String line : trainClean.split("\n")) { if (line.isEmpty() || shown++ >= 10) break; System.out.println("  " + line); }

        int valChars = Math.min(20000, testClean.length());
        System.out.println("\nTrain with (about 10 minutes):");
        System.out.printf("java ChatTransformer data=%s chars=%d valstart=%d valchars=%d vocab=512 context=96 seconds=600 threads=8 name=talk%n", out, cut, valStart, valChars);
        System.out.printf("Then talk to it: java DialogueChat name=talk data=%s chars=%d%n", out, cut);
    }

    // ---------- turning rows into text ----------

    /** Each row becomes "User: ...\nBot: ...\n" lines; rows without at least one user line and one bot line are skipped. */
    static List<String> conversations(List<Object> rows, String what) {
        List<String> result = new ArrayList<>();
        int skipped = 0;
        for (Object row : rows) {
            List<?> messages = findMessages(row);
            if (messages == null) { skipped++; continue; }
            StringBuilder sb = new StringBuilder();
            int users = 0, bots = 0;
            for (Object m : messages) {
                Map<?, ?> msg = (Map<?, ?>) m;
                String role = String.valueOf(msg.get("role")).toLowerCase(), label;
                if (role.equals("user")) { label = "User"; users++; }
                else if (role.equals("assistant")) { label = "Bot"; bots++; }
                else continue;                                            // system messages and anything else are left out
                String text = String.valueOf(msg.get("content")).replaceAll("\\s+", " ").trim();
                if (text.isEmpty()) continue;
                sb.append(label).append(": ").append(text).append('\n');
            }
            if (users == 0 || bots == 0) { skipped++; continue; }
            result.add(sb.toString());
        }
        if (result.isEmpty() && !rows.isEmpty()) {
            Object first = rows.get(0);
            System.out.println("Could not find a list of {role, content} messages in the " + what + " rows. The first row looks like: " + String.valueOf(first).substring(0, Math.min(600, String.valueOf(first).length())));
        }
        if (skipped > 0) System.out.println("(" + what + ": skipped " + skipped + " rows without a usable conversation.)");
        return result;
    }

    /** The first list anywhere inside the row whose items all look like {"role": text, "content": text}. */
    static List<?> findMessages(Object node) {
        if (node instanceof List) {
            List<?> list = (List<?>) node;
            boolean all = !list.isEmpty();
            for (Object o : list) all &= o instanceof Map && ((Map<?, ?>) o).get("role") instanceof String && ((Map<?, ?>) o).get("content") instanceof String;
            if (all) return list;
            for (Object o : list) { List<?> r = findMessages(o); if (r != null) return r; }
        } else if (node instanceof Map) {
            for (Object o : ((Map<?, ?>) node).values()) { List<?> r = findMessages(o); if (r != null) return r; }
        }
        return null;
    }

    // ---------- getting the rows ----------

    static List<Object> readLines(String file) throws IOException {
        List<Object> rows = new ArrayList<>();
        for (String line : Files.readAllLines(Paths.get(file), StandardCharsets.UTF_8)) if (!line.trim().isEmpty()) rows.add(Json.parse(line));
        System.out.println("Read " + rows.size() + " rows from " + file);
        return rows;
    }

    /** Downloads one split, 100 rows per request, and returns the rows (each one is the map of column name to value). */
    static List<Object> fetchSplit(HttpClient client, String base, String dataset, String config, String split) throws Exception {
        List<Object> rows = new ArrayList<>();
        int offset = 0, total = Integer.MAX_VALUE;
        while (offset < total) {
            String url = base + "/rows?dataset=" + URLEncoder.encode(dataset, "UTF-8") + "&config=" + URLEncoder.encode(config, "UTF-8")
                + "&split=" + URLEncoder.encode(split, "UTF-8") + "&offset=" + offset + "&length=" + PAGE;
            Map<?, ?> page = (Map<?, ?>) Json.parse(get(client, url));
            if (page.get("num_rows_total") instanceof Double) total = ((Double) page.get("num_rows_total")).intValue();
            List<?> got = page.get("rows") instanceof List ? (List<?>) page.get("rows") : new ArrayList<>();
            if (got.isEmpty()) break;
            for (Object r : got) rows.add(r instanceof Map && ((Map<?, ?>) r).get("row") != null ? ((Map<?, ?>) r).get("row") : r);
            offset += got.size();
            System.out.printf("\r%s: %,d of %,d rows", split, rows.size(), total == Integer.MAX_VALUE ? rows.size() : total);
            Thread.sleep(150);                                           // be gentle with the free service
        }
        System.out.println();
        return rows;
    }

    static String get(HttpClient client, String url) throws Exception {
        for (int attempt = 1; ; attempt++) {
            HttpRequest req = HttpRequest.newBuilder(URI.create(url)).header("User-Agent", "mini-llm-java").GET().build();
            HttpResponse<String> resp = null;
            IOException failure = null;
            try {
                resp = client.send(req, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            } catch (IOException e) {
                failure = e;
            }
            if (resp != null && resp.statusCode() == 200) return resp.body();
            boolean worthRetrying = resp == null || resp.statusCode() == 429 || resp.statusCode() >= 500;   // a 404 or similar will not get better
            if (!worthRetrying || attempt >= 4) {
                if (resp == null) throw failure;
                throw new IOException("The server answered " + resp.statusCode() + " for " + url + ": " + resp.body().substring(0, Math.min(300, resp.body().length())));
            }
            Thread.sleep(2000L * attempt);
        }
    }

    // ---------- a small JSON reader (objects, lists, strings, numbers, true/false/null) ----------

    static final class Json {
        final String s;
        int i = 0;

        Json(String s) { this.s = s; }

        static Object parse(String text) {
            Json j = new Json(text);
            j.ws();
            Object v = j.value();
            j.ws();
            if (j.i != text.length()) throw j.err("extra text after the value");
            return v;
        }

        RuntimeException err(String what) { return new IllegalArgumentException("JSON problem at character " + i + ": " + what); }

        void ws() { while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++; }

        void expect(char c) { if (i >= s.length() || s.charAt(i) != c) throw err("expected '" + c + "'"); i++; }

        Object value() {
            if (i >= s.length()) throw err("unexpected end");
            char c = s.charAt(i);
            if (c == '{') return object();
            if (c == '[') return array();
            if (c == '"') return string();
            if (s.startsWith("true", i)) { i += 4; return Boolean.TRUE; }
            if (s.startsWith("false", i)) { i += 5; return Boolean.FALSE; }
            if (s.startsWith("null", i)) { i += 4; return null; }
            int start = i;
            while (i < s.length() && "+-0123456789.eE".indexOf(s.charAt(i)) >= 0) i++;
            if (start == i) throw err("unexpected character '" + c + "'");
            return Double.valueOf(s.substring(start, i));
        }

        Map<String, Object> object() {
            expect('{');
            Map<String, Object> map = new LinkedHashMap<>();
            ws();
            if (i < s.length() && s.charAt(i) == '}') { i++; return map; }
            while (true) {
                ws();
                String key = string();
                ws();
                expect(':');
                ws();
                map.put(key, value());
                ws();
                if (i >= s.length()) throw err("unexpected end inside an object");
                char c = s.charAt(i++);
                if (c == '}') return map;
                if (c != ',') throw err("expected ',' or '}'");
            }
        }

        List<Object> array() {
            expect('[');
            List<Object> list = new ArrayList<>();
            ws();
            if (i < s.length() && s.charAt(i) == ']') { i++; return list; }
            while (true) {
                ws();
                list.add(value());
                ws();
                if (i >= s.length()) throw err("unexpected end inside a list");
                char c = s.charAt(i++);
                if (c == ']') return list;
                if (c != ',') throw err("expected ',' or ']'");
            }
        }

        String string() {
            expect('"');
            StringBuilder sb = new StringBuilder();
            while (true) {
                if (i >= s.length()) throw err("unterminated string");
                char c = s.charAt(i++);
                if (c == '"') return sb.toString();
                if (c != '\\') { sb.append(c); continue; }
                if (i >= s.length()) throw err("unterminated escape");
                char e = s.charAt(i++);
                switch (e) {
                    case '"': sb.append('"'); break;
                    case '\\': sb.append('\\'); break;
                    case '/': sb.append('/'); break;
                    case 'b': sb.append('\b'); break;
                    case 'f': sb.append('\f'); break;
                    case 'n': sb.append('\n'); break;
                    case 'r': sb.append('\r'); break;
                    case 't': sb.append('\t'); break;
                    case 'u':
                        if (i + 4 > s.length()) throw err("short \\u escape");
                        sb.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                        i += 4;
                        break;
                    default: throw err("unknown escape \\" + e);
                }
            }
        }
    }
}
