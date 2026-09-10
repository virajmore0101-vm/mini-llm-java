import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TrainDigits {

    public static void main(String[] args) throws IOException {
        List<Vector> trainInputs = new ArrayList<>();
        List<Vector> trainTargets = new ArrayList<>();
        loadCsv("mnist_train_100.csv", trainInputs, trainTargets);

        List<Vector> testInputs = new ArrayList<>();
        List<Vector> testTargets = new ArrayList<>();
        loadCsv("mnist_test_10.csv", testInputs, testTargets);

        System.out.println("Loaded " + trainInputs.size() + " training images, "
            + testInputs.size() + " test images.");

        NeuralNetwork net = new NeuralNetwork(784, 100, 10, 0.2);

        int epochs = 1000;
        long start = System.currentTimeMillis();
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < trainInputs.size(); i++) {
                net.trainStep(trainInputs.get(i), trainTargets.get(i));
            }
            if (epoch % 200 == 0) {
                System.out.println("Epoch " + epoch + "...");
            }
        }
        System.out.println("Training done in " + (System.currentTimeMillis() - start) + " ms.");

        System.out.println("\nTesting on " + testInputs.size() + " images the network never trained on:");
        int correct = 0;
        for (int i = 0; i < testInputs.size(); i++) {
            Vector output = net.predict(testInputs.get(i));
            int predicted = argMax(output);
            int actual = argMax(testTargets.get(i));
            System.out.println("Predicted: " + predicted + "   Actual: " + actual
                + (predicted == actual ? "   correct" : "   WRONG"));
            if (predicted == actual) correct++;
        }

        System.out.printf("%nAccuracy: %d/%d (%.1f%%)%n",
            correct, testInputs.size(), 100.0 * correct / testInputs.size());
    }

    private static void loadCsv(String path, List<Vector> inputs, List<Vector> targets) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) continue;
                String[] parts = line.split(",");
                int label = Integer.parseInt(parts[0]);

                double[] pixels = new double[784];
                for (int i = 0; i < 784; i++) {
                    pixels[i] = (Double.parseDouble(parts[i + 1]) / 255.0 * 0.99) + 0.01;
                }
                inputs.add(new Vector(pixels));

                double[] target = new double[10];
                for (int i = 0; i < 10; i++) target[i] = 0.01;
                target[label] = 0.99;
                targets.add(new Vector(target));
            }
        }
    }

    private static int argMax(Vector v) {
        int best = 0;
        for (int i = 1; i < v.size(); i++) {
            if (v.get(i) > v.get(best)) best = i;
        }
        return best;
    }
}
