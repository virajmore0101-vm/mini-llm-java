/**
 * A simple vector of doubles, built from scratch (no external libraries).
 * This is the foundation everything else (neurons, layers, models) is built on.
 */
public class Vector {

    private final double[] values;

    public Vector(int size) {
        this.values = new double[size];
    }

    public Vector(double[] values) {
        this.values = values;
    }

    public int size() {
        return values.length;
    }

    public double get(int i) {
        return values[i];
    }

    public void set(int i, double v) {
        values[i] = v;
    }

    public Vector add(Vector other) {
        checkSameSize(other);
        Vector result = new Vector(size());
        for (int i = 0; i < size(); i++) {
            result.set(i, this.get(i) + other.get(i));
        }
        return result;
    }

    public Vector subtract(Vector other) {
        checkSameSize(other);
        Vector result = new Vector(size());
        for (int i = 0; i < size(); i++) {
            result.set(i, this.get(i) - other.get(i));
        }
        return result;
    }

    public Vector scale(double scalar) {
        Vector result = new Vector(size());
        for (int i = 0; i < size(); i++) {
            result.set(i, this.get(i) * scalar);
        }
        return result;
    }

    public double dot(Vector other) {
        checkSameSize(other);
        double sum = 0;
        for (int i = 0; i < size(); i++) {
            sum += this.get(i) * other.get(i);
        }
        return sum;
    }

    public Vector multiplyElementwise(Vector other) {
        checkSameSize(other);
        Vector result = new Vector(size());
        for (int i = 0; i < size(); i++) {
            result.set(i, this.get(i) * other.get(i));
        }
        return result;
    }

    private void checkSameSize(Vector other) {
        if (this.size() != other.size()) {
            throw new IllegalArgumentException(
                "Vector size mismatch: " + this.size() + " vs " + other.size());
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {
            sb.append(String.format("%.3f", values[i]));
            if (i < values.length - 1) sb.append(", ");
        }
        return sb.append("]").toString();
    }
}
