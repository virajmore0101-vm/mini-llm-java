import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;
import java.util.Scanner;

/**
 * The finish line: wraps the character-level model from step 4 in an
 * interactive loop. Type something, press enter, get generated text back.
 */
public class Chat {

    public static void main(String[] args) throws IOException {
        String text = Files.readString(Paths.get("shakespeare.txt"));
        int contextLength = 6;

        System.out.println("Training on " + text.length() + " characters of Shakespeare... (a second or two)");
        CharModel model = new CharModel(contextLength);
        model.train(text);
        System.out.println("Ready. Type something and press enter (or type 'exit' to quit):\n");

        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("You: ");
            if (!scanner.hasNextLine()) break;
            String input = scanner.nextLine();

            if (input.equalsIgnoreCase("exit")) {
                System.out.println("Goodbye!");
                break;
            }

            String seed = findSeed(model, text, input, contextLength);
            String generated = model.generate(seed, 300);
            System.out.println("\nModel: " + generated + "\n");
        }
        scanner.close();
    }

    private static String findSeed(CharModel model, String text, String input, int contextLength) {
        if (input.length() >= contextLength) {
            String candidate = input.substring(input.length() - contextLength);
            if (model.knowsContext(candidate)) {
                return candidate;
            }
        }
        System.out.println("(That exact phrase wasn't in the training text -- improvising from a similar starting point.)");
        int randomStart = new Random().nextInt(text.length() - contextLength);
        return text.substring(randomStart, randomStart + contextLength);
    }
}
