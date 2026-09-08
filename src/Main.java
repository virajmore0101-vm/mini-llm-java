public class Main {
    public static void main(String[] args) {

        System.out.println("=== Vector basics ===");
        Vector a = new Vector(new double[]{1, 2, 3});
        Vector b = new Vector(new double[]{4, 5, 6});
        System.out.println("a = " + a);
        System.out.println("b = " + b);
        System.out.println("a + b = " + a.add(b));
        System.out.println("a . b (dot product) = " + a.dot(b));

        System.out.println("\n=== A tiny 'layer' of 2 neurons, each taking 3 inputs ===");
        Matrix weights = new Matrix(new double[][]{
            {0.2, 0.8, -0.5},
            {0.5, -0.4, 0.3}
        });
        Vector input = new Vector(new double[]{1.0, 0.5, -1.5});

        System.out.println("weights =\n" + weights);
        System.out.println("input = " + input);

        Vector output = weights.multiply(input);
        System.out.println("output (weights * input) = " + output);
        System.out.println("-> neuron 1 fired: " + output.get(0));
        System.out.println("-> neuron 2 fired: " + output.get(1));
    }
}
