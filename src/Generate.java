import java.nio.file.*;
import java.util.*;

public class Generate {
    public static void main(String[] args) throws Exception {
        String text = Files.readString(Paths.get("shakespeare.txt")).substring(0, 2000);
        int T = 8;
        MiniTransformer m = new MiniTransformer(text, T, 16, 0.03);
        for (int e = 0; e < 150; e++)
            for (int i = 0; i + T < text.length(); i++)
                m.trainStep(text.substring(i, i + T), text.charAt(i + T));

        Random r = new Random();
        StringBuilder out = new StringBuilder(text.substring(0, T));
        for (int i = 0; i < 150; i++) out.append(m.sampleNext(out.substring(out.length() - T), r));
        System.out.println(out);
    }
}
