/**
 * A simple 2D matrix of doubles, built from scratch.
 * Rows = one row per neuron's weights. Columns = one column per input.
 */
public class Matrix {

    private final double[][] values;
    private final int rows;
    private final int cols;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.values = new double[rows][cols];
    }

    public Matrix(double[][] values) {
        this.values = values;
        this.rows = values.length;
        this.cols = values[0].length;
    }

    public int rows() { return rows; }
    public int cols() { return cols; }

    public double get(int r, int c) { return values[r][c]; }
    public void set(int r, int c, double v) { values[r][c] = v; }

    public Vector multiply(Vector v) {
        if (this.cols != v.size()) {
            throw new IllegalArgumentException(
                "Matrix has " + cols + " columns but vector has " + v.size() + " elements");
        }
        Vector result = new Vector(this.rows);
        for (int r = 0; r < rows; r++) {
            double sum = 0;
            for (int c = 0; c < cols; c++) {
                sum += this.get(r, c) * v.get(c);
            }
            result.set(r, sum);
        }
        return result;
    }

    public Matrix multiply(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException(
                "Cannot multiply " + rows + "x" + cols + " by " + other.rows + "x" + other.cols);
        }
        Matrix result = new Matrix(this.rows, other.cols);
        for (int r = 0; r < this.rows; r++) {
            for (int c = 0; c < other.cols; c++) {
                double sum = 0;
                for (int k = 0; k < this.cols; k++) {
                    sum += this.get(r, k) * other.get(k, c);
                }
                result.set(r, c, sum);
            }
        }
        return result;
    }

    public Matrix transpose() {
        Matrix result = new Matrix(this.cols, this.rows);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result.set(c, r, this.get(r, c));
            }
        }
        return result;
    }

    public Matrix add(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix size mismatch");
        }
        Matrix result = new Matrix(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result.set(r, c, this.get(r, c) + other.get(r, c));
            }
        }
        return result;
    }

    public Matrix scale(double scalar) {
        Matrix result = new Matrix(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result.set(r, c, this.get(r, c) * scalar);
            }
        }
        return result;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int r = 0; r < rows; r++) {
            sb.append("[");
            for (int c = 0; c < cols; c++) {
                sb.append(String.format("%.3f", values[r][c]));
                if (c < cols - 1) sb.append(", ");
            }
            sb.append("]\n");
        }
        return sb.toString();
    }
}
