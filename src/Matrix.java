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
            double[] row = values[r];
            double sum = 0;
            for (int c = 0; c < cols; c++) sum += row[c] * v.get(c);
            result.set(r, sum);
        }
        return result;
    }

    /** this * other. Loops ordered i-k-j so memory is read row by row (much friendlier to the CPU cache). */
    public Matrix multiply(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException(
                "Cannot multiply " + rows + "x" + cols + " by " + other.rows + "x" + other.cols);
        }
        Matrix result = new Matrix(this.rows, other.cols);
        double[][] a = values, b = other.values, c = result.values;
        int n = this.cols, m = other.cols;
        for (int i = 0; i < rows; i++) {
            double[] ai = a[i], ci = c[i];
            for (int k = 0; k < n; k++) {
                double aik = ai[k];
                double[] bk = b[k];
                for (int j = 0; j < m; j++) ci[j] += aik * bk[j];
            }
        }
        return result;
    }

    /** this * other^T, without ever building the transposed copy. */
    public Matrix multiplyByTranspose(Matrix other) {
        if (this.cols != other.cols) {
            throw new IllegalArgumentException(
                "Cannot multiply " + rows + "x" + cols + " by transpose of " + other.rows + "x" + other.cols);
        }
        Matrix result = new Matrix(this.rows, other.rows);
        double[][] a = values, b = other.values, c = result.values;
        for (int i = 0; i < rows; i++) {
            double[] ai = a[i], ci = c[i];
            for (int j = 0; j < other.rows; j++) {
                double[] bj = b[j];
                double sum = 0;
                for (int k = 0; k < cols; k++) sum += ai[k] * bj[k];
                ci[j] = sum;
            }
        }
        return result;
    }

    /** this^T * other, without ever building the transposed copy. */
    public Matrix transposeMultiply(Matrix other) {
        if (this.rows != other.rows) {
            throw new IllegalArgumentException(
                "Cannot multiply transpose of " + rows + "x" + cols + " by " + other.rows + "x" + other.cols);
        }
        Matrix result = new Matrix(this.cols, other.cols);
        double[][] a = values, b = other.values, c = result.values;
        for (int k = 0; k < rows; k++) {
            double[] ak = a[k], bk = b[k];
            for (int i = 0; i < cols; i++) {
                double aki = ak[i];
                double[] ci = c[i];
                for (int j = 0; j < other.cols; j++) ci[j] += aki * bk[j];
            }
        }
        return result;
    }

    public static Matrix outerProduct(Vector a, Vector b) {
        Matrix result = new Matrix(a.size(), b.size());
        for (int i = 0; i < a.size(); i++) {
            double ai = a.get(i);
            double[] row = result.values[i];
            for (int j = 0; j < b.size(); j++) row[j] = ai * b.get(j);
        }
        return result;
    }

    public Matrix transpose() {
        Matrix result = new Matrix(this.cols, this.rows);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) result.values[c][r] = values[r][c];
        }
        return result;
    }

    public Matrix add(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix size mismatch");
        }
        Matrix result = new Matrix(rows, cols);
        for (int r = 0; r < rows; r++) {
            double[] a = values[r], b = other.values[r], c = result.values[r];
            for (int j = 0; j < cols; j++) c[j] = a[j] + b[j];
        }
        return result;
    }

    public Matrix scale(double scalar) {
        Matrix result = new Matrix(rows, cols);
        for (int r = 0; r < rows; r++) {
            double[] a = values[r], c = result.values[r];
            for (int j = 0; j < cols; j++) c[j] = a[j] * scalar;
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
