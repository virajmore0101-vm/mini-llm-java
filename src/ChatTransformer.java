import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;
import java.util.Scanner;

public class ChatTransformer {
    public static void main(String[] args) throws IOException {
        String text = Files.readString(Paths.get("shakespeare.txt")).substring(0, 20000);
        int T = 10;

        System.out.println("Training on " + text.length() + " characters (3 layers, 4 heads, 24-dim)...");
        System.out.println("This takes roughly 15-20 minutes. It's working the whole time, not stuck --");
        System.out.println("the epoch counter below will update every few minutes to prove it's alive.\n");

        MultiLayerTransformer model = new MultiLayerTransformer(text, T, 24, 3, 4, 0.004);
        long start = System.currentTimeMillis();
        int epochs = 55;
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i + T < text.length(); i++) {
                model.trainStep(text.substring(i, i + T), text.charAt(i + T));
            }
            if (epoch % 5 == 0) {
                long elapsed = (System.currentTimeMillis() - start) / 1000;
                System.out.println("  ...epoch " + epoch + "/" + epochs + " (" + elapsed + "s elapsed)");
            }
        }
        System.out.println("Training complete in " + (System.currentTimeMillis() - start) / 1000 + "s. Ready!");
        System.out.println("Type something and press enter (or 'exit' to quit):\n");

        Random rand = new Random();
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("You: ");
            if (!scanner.hasNextLine()) break;
            String input = scanner.nextLine();
            if (input.equalsIgnoreCase("exit")) {
                System.out.println("Goodbye!");
                break;
            }

            String seed = findSeed(model, text, input, T, rand);
            StringBuilder out = new StringBuilder(seed);
            for (int i = 0; i < 250; i++) {
                out.append(model.sampleNext(out.substring(out.length() - T), rand));
            }
            System.out.println("\nModel: " + out + "\n");
        }
        scanner.close();
    }

    private static String findSeed(MultiLayerTransformer model, String text, String input, int T, Random rand) {
        if (input.length() >= T) {
            String candidate = input.substring(input.length() - T);
            boolean known = true;
            for (char c : candidate.toCharArray()) {
                if (!model.charToIndex.containsKey(c)) { known = false; break; }
            }
            if (known) return candidate;
        }
        System.out.println("(Your exact text wasn't all in the training vocabulary -- starting from a real snippet instead.)");
        int randomStart = rand.nextInt(text.length() - T);
        return text.substring(randomStart, randomStart + T);
    }
}
