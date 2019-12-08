import com.google.gson.JsonObject;

import java.util.Arrays;

public class DropoutLayer extends Layer {
    private final boolean[] dropped;
    private double dropProb;
    private Vol in_act;

    DropoutLayer() {
        this(new LayerConfig());
    }

    DropoutLayer(final LayerConfig opt) {
        this.out_sx = opt.getOutSX();
        this.out_sy = opt.getOutSY();
        this.out_depth = opt.getOutDepth();
        this.type = LayerType.DROPOUT;
        this.dropProb = LayerConfig.getOrDefault(0.5, opt.getDropProb());
        this.dropped = new boolean[this.out_sx * this.out_sy * this.out_depth];
        Arrays.fill(this.dropped, false);
    }

    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        this.in_act = vol;
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
        this.out_act = vol2;
        return this.out_act;
    }

    @Override
    public ParamsAndGrads[] getParamsAndGrads() {
        return new ParamsAndGrads[0];
    }

    @Override
    public JsonObject toJSON() {
        final JsonObject json = new JsonObject();
        json.addProperty("out_depth", this.out_depth);
        json.addProperty("out_sx", this.out_sx);
        json.addProperty("out_sy", this.out_sy);
        json.addProperty("type", this.type.toString());
        json.addProperty("dropProb", this.dropProb);
        return json;
    }

    @Override
    public void fromJSON(final JsonObject json) {
        this.out_depth = json.get("out_depth").getAsInt();
        this.out_sx = json.get("out_sx").getAsInt();
        this.out_sy = json.get("out_sy").getAsInt();
        this.type = LayerType.valueOf(json.get("type").getAsString());
        this.dropProb = json.get("dropProb").getAsDouble();
    }

    @Override
    public double backward(final Vol output) {
        final Vol vol = this.in_act;
        final Vol chainGrad = this.out_act;
        vol.dw = Utils.zerosDouble(vol.w.length);
        for (int i = 0; i < vol.w.length; i++) {
            if (!this.dropped[i]) {
                vol.dw[i] = chainGrad.dw[i];
            }
        }
        return 0;
    }
}
