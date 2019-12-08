import com.google.gson.JsonObject;

import java.util.Arrays;

public class DropoutLayer extends Layer {
    private final boolean[] dropped;
    private double dropProb;

    DropoutLayer() {
        this(new LayerConfig());
    }

    DropoutLayer(final LayerConfig opt) {
        this.outSX = opt.getOutSX();
        this.outSY = opt.getOutSY();
        this.outDepth = opt.getOutDepth();
        this.type = LayerType.DROPOUT;
        this.dropProb = LayerConfig.getOrDefault(0.5, opt.getDropProb());
        this.dropped = new boolean[this.outSX * this.outSY * this.outDepth];
        Arrays.fill(this.dropped, false);
    }

    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        this.inAct = vol;
        final Vol vol2 = vol.clone();

        if (isTraining) {
            for (int i = 0; i < vol.w.length; i++) {
                if (Math.random() < this.dropProb) {
                    vol2.w[i] = 0;
                    this.dropped[i] = true;
                } else {
                    this.dropped[i] = false;
                }
            }
        } else {
            for (int i = 0; i < vol2.w.length; i++) {
                vol2.w[i] *= this.dropProb;
            }
        }
        this.outAct = vol2;
        return this.outAct;
    }

    @Override
    public ParamsAndGrads[] getParamsAndGrads() {
        return new ParamsAndGrads[0];
    }

    @Override
    public JsonObject toJSON() {
        final JsonObject json = new JsonObject();
        json.addProperty("outDepth", this.outDepth);
        json.addProperty("outSX", this.outSX);
        json.addProperty("outSY", this.outSY);
        json.addProperty("type", this.type.toString());
        json.addProperty("dropProb", this.dropProb);
        return json;
    }

    @Override
    public void fromJSON(final JsonObject json) {
        this.outDepth = json.get("outDepth").getAsInt();
        this.outSX = json.get("outSX").getAsInt();
        this.outSY = json.get("outSY").getAsInt();
        this.type = LayerType.valueOf(json.get("type").getAsString());
        this.dropProb = json.get("dropProb").getAsDouble();
    }

    @Override
    public double backward(final Vol output) {
        final Vol vol = this.inAct;
        final Vol chainGrad = this.outAct;
        vol.dw = Utils.zerosDouble(vol.w.length);
        for (int i = 0; i < vol.w.length; i++) {
            if (!this.dropped[i]) {
                vol.dw[i] = chainGrad.dw[i];
            }
        }
        return 0;
    }
}
