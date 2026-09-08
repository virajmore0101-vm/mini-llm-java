public class TrainNeuron {
    public static void main(String[] args) {

        Vector[] inputs = {
            new Vector(new double[]{0, 0}),
            new Vector(new double[]{0, 1}),
            new Vector(new double[]{1, 0}),
            new Vector(new double[]{1, 1})
        };
        double[] targets = {0, 0, 0, 1};

        Neuron neuron = new Neuron(2, 0.5);

        int epochs = 10000;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;
            for (int i = 0; i < inputs.length; i++) {
                neuron.trainStep(inputs[i], targets[i]);
                double prediction = neuron.predict(inputs[i]);
                totalError += Math.pow(targets[i] - prediction, 2);
            }
            if (epoch % 2000 == 0) {
                System.out.printf("Epoch %d, total squared error: %.5f%n", epoch, totalError);
            }
        }

        System.out.println("\nFinal predictions (AND gate):");
        for (int i = 0; i < inputs.length; i++) {
            double prediction = neuron.predict(inputs[i]);
            System.out.printf("%s -> %.4f (target %.0f)%n", inputs[i], prediction, targets[i]);
        }

        System.out.println("\nLearned weights: " + neuron.getWeights());
        System.out.printf("Learned bias: %.4f%n", neuron.getBias());
    }
}
