package layers;

import com.google.gson.JsonObject;
import utils.ParamsAndGrads;
import utils.Utils;
import utils.Vol;

public class LocalResponseNormalizationLayer extends Layer {
    private int k;
    private int n;
    private double alpha;
    private double beta;
    private Vol S;

    public LocalResponseNormalizationLayer() {
        this(new LayerConfig());
    }

    public LocalResponseNormalizationLayer(final LayerConfig opt) {
        //required
        this.k = opt.getK();
        this.n = opt.getN();
        this.alpha = opt.getAlpha();
        this.beta = opt.getBeta();

        //computed
        this.outSX = opt.getInSX();
        this.outSY = opt.getInSY();
        this.outDepth = opt.getInDepth();
        this.type = LayerType.LRN;

        if (this.n % 2 == 0) {
            throw new RuntimeException("WARNING: n should be odd for LRN layer");
        }
    }

    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        this.inAct = vol;

        final Vol A = vol.cloneAndZero();
        this.S = vol.cloneAndZero();
        final int n2 = (int) Math.floor((double) this.n / 2);
        for (int x = 0; x < vol.sx; x++) {
            for (int y = 0; y < vol.sy; y++) {
                for (int i = 0; i < vol.depth; i++) {
                    final double ai = vol.get(x, y, i);
                    // normalize in a window of size n
                    double den = 0;
                    for (int j = Math.max(0, i - n2); j <= Math.min(i + n2, vol.depth - 1); j++) {
                        final double aa = vol.get(x, y, j);
                        den += aa * aa;
                    }
                    den *= this.alpha / this.n;
                    den += this.k;
                    this.S.set(x, y, i, den); // will be useful for backprop
                    den = Math.pow(den, this.beta);
                    A.set(x, y, i, ai / den);
                }
            }
        }

        this.outAct = A;
        return this.outAct; // dummy identity function for now
    }

    @Override
    public ParamsAndGrads[] getParamsAndGrads() {
        return new ParamsAndGrads[0];
    }


    @Override
    public JsonObject toJSON() {
        final JsonObject json = new JsonObject();
        json.addProperty("k", this.k);
        json.addProperty("n", this.n);
        json.addProperty("alpha", this.alpha);
        json.addProperty("beta", this.beta);
        json.addProperty("outDepth", this.outDepth);
        json.addProperty("outSX", this.outSX);
        json.addProperty("outSY", this.outSY);
        json.addProperty("type", this.type.toString());
        return json;
    }

    @Override
    public void fromJSON(final JsonObject json) {
        this.k = json.get("k").getAsInt();
        this.n = json.get("n").getAsInt();
        this.alpha = json.get("alpha").getAsDouble(); // normalize by size
        this.beta = json.get("beta").getAsDouble();
        this.outDepth = json.get("outDepth").getAsInt();
        this.outSX = json.get("outSX").getAsInt();
        this.outSY = json.get("outSY").getAsInt();
        this.type = LayerType.valueOf(json.get("type").getAsString());
    }

    @Override
    public double backward(final Vol output) {
        // evaluate gradient wrt data
        final Vol vol = this.inAct; // we need to set dw of this
        vol.dw = Utils.zerosDouble(vol.w.length); // zero out gradient wrt data

        final int n2 = (int) Math.floor((double) this.n / 2);
        for (int x = 0; x < vol.sx; x++) {
            for (int y = 0; y < vol.sy; y++) {
                for (int i = 0; i < vol.depth; i++) {

                    final double chainGrad = this.outAct.getGrad(x, y, i);
                    final double S = this.S.get(x, y, i);
                    final double SB = Math.pow(S, this.beta);
                    final double SB2 = SB * SB;

                    // normalize in a window of size n
                    for (int j = Math.max(0, i - n2); j <= Math.min(i + n2, vol.depth - 1); j++) {
                        final double aj = vol.get(x, y, j);
                        double g = -aj * this.beta * Math.pow(S, this.beta - 1) * this.alpha / this.n * 2 * aj;
                        if (j == i) {
                            g += SB;
                        }
                        g /= SB2;
                        g *= chainGrad;
                        vol.addGrad(x, y, j, g);
                    }
                }
            }
        }
        return 0;
    }
}
