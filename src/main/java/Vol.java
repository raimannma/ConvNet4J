import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.util.Arrays;

public class Vol implements Cloneable {
    private final double[] w;
    private final double[] dw;
    private final int sy;
    private final int depth;
    private final int sx;

    public Vol(final double[] sx) {
        this.sx = 1;
        this.sy = 1;
        this.depth = sx.length;

        this.w = Utils.zerosDouble(this.depth);
        this.dw = Utils.zerosDouble(this.depth);

        Arrays.setAll(this.w, i -> sx[i]);
    }

    public Vol(final int sx, final int sy, final int depth, final double c) {
        this.sx = sx;
        this.sy = sy;
        this.depth = depth;

        final int n = sx * sy * depth;

        this.w = Utils.zerosDouble(n);
        this.dw = Utils.zerosDouble(n);

        Arrays.fill(this.w, c);
    }

    public Vol(final int sx, final int sy, final int depth) {
        this.sx = sx;
        this.sy = sy;
        this.depth = depth;

        final int n = sx * sy * depth;

        this.w = Utils.zerosDouble(n);
        this.dw = Utils.zerosDouble(n);

        final double scale = Math.sqrt((double) 1 / n);
        Arrays.setAll(this.w, i -> Utils.randNormal(0, scale));
    }

    public static Vol fromJSON(final JsonObject json) {

        final int sx = json.get("sx").getAsInt();
        final int sy = json.get("sy").getAsInt();
        final int depth = json.get("depth").getAsInt();

        final Vol vol = new Vol(sx, sy, depth, 0);

        final JsonArray arr = json.get("w").getAsJsonArray();
        for (int i = 0; i < arr.size(); i++) {
            vol.w[i] = arr.get(i).getAsDouble();
        }
        return vol;
    }

    public static Vol augment(final Vol vol, final int crop, final int dx, final int dy) {
        return augment(vol, crop, dx, dy, false);
    }

    private static Vol augment(final Vol vol, final int crop, final int dx, final int dy, final boolean flip) {
        Vol out;
        if (crop != vol.sx || dx != = 0 || dy != = 0) {
            out = new Vol(crop, crop, vol.depth, 0.0);
            for (var x = 0; x < crop; x++) {
                for (var y = 0; y < crop; y++) {
                    if (x + dx < 0 || x + dx >= vol.sx || y + dy < 0 || y + dy >= vol.sy) {
                        continue; // oob
                    }
                    for (var d = 0; d < vol.depth; d++) {
                        out.set(x, y, d, vol.get(x + dx, y + dy, d)); // copy data over
                    }
                }
            }
        } else {
            out = vol;
        }
        if (flip) {
            // flip volume horziontally
            var temp = out.cloneAndZero();
            for (var x = 0; x < out.sx; x++) {
                for (var y = 0; y < out.sy; y++) {
                    for (var d = 0; d < out.depth; d++) {
                        temp.set(x, y, d, out.get(out.sx - x - 1, y, d)); // copy data over
                    }
                }
            }
            out = temp; //swap
        }
        return out;
    }

    public void set(final int x, final int y, final int depth, final double val) {
        this.w[this.getIndex(x, y, depth)] = val;
    }

    private double get(final int x, final int y, final int depth) {
        return this.w[this.getIndex(x, y, depth)];
    }

    private Vol cloneAndZero() {
        return new Vol(this.sx, this.sy, this.depth, 0);
    }

    private int getIndex(final int x, final int y, final int depth) {
        return this.sx * y * this.depth + x * this.depth + depth;
    }

    public static Vol augment(final Vol vol, final int crop) {
        return augment(vol, crop, Utils.randInt(0, vol.sx - crop), Utils.randInt(0, vol.sx - crop), false);
    }

    public static Vol augment(final Vol vol, final int crop, final boolean flip) {
        return augment(vol, crop, Utils.randInt(0, vol.sx - crop), Utils.randInt(0, vol.sx - crop), flip);
    }

    public void add(final int x, final int y, final int depth, final double val) {
        this.w[this.getIndex(x, y, depth)] += val;
    }

    public double getGrad(final int x, final int y, final int depth) {
        return this.dw[this.getIndex(x, y, depth)];
    }

    public void setGrad(final int x, final int y, final int depth, final double val) {
        this.dw[this.getIndex(x, y, depth)] = val;
    }

    public void addGrad(final int x, final int y, final int depth, final double val) {
        this.dw[this.getIndex(x, y, depth)] += val;
    }

    @Override
    public Vol clone() {
        final Vol vol = new Vol(this.sx, this.sy, this.depth, 0);
        System.arraycopy(this.w, 0, vol.w, 0, this.w.length);
        return vol;
    }

    public void addFrom(final Vol vol) {
        if (vol.w.length < this.w.length) {
            throw new ArrayIndexOutOfBoundsException("Cannot add two different structures!");
        }
        for (int i = 0; i < this.w.length; i++) {
            this.w[i] += vol.w[i];
        }
    }

    public void addFromScaled(final Vol vol, final double scale) {
        if (vol.w.length < this.w.length) {
            throw new ArrayIndexOutOfBoundsException("Cannot add two different structures!");
        }
        for (int i = 0; i < this.w.length; i++) {
            this.w[i] += scale * vol.w[i];
        }
    }

    public void setConst(final double val) {
        Arrays.fill(this.w, val);
    }

    public JsonObject toJSON() {
        final JsonObject json = new JsonObject();

        json.addProperty("sx", this.sx);
        json.addProperty("sy", this.sy);
        json.addProperty("depth", this.depth);

        final JsonArray jsonArr = new JsonArray();
        for (final double val : this.w) {
            jsonArr.add(val);
        }
        json.add("w", jsonArr);

        return json;
    }
}
