package layers;

import com.google.gson.JsonObject;
import utils.ParamsAndGrads;
import utils.Utils;
import utils.Vol;

public class SigmoidLayer extends Layer {
    public SigmoidLayer() {
        this(new LayerConfig());
    }

    public SigmoidLayer(final LayerConfig opt) {
        this.outSX = opt.getInSX();
        this.outSY = opt.getInSY();
        this.outDepth = opt.getInDepth();
        this.type = Layer.LayerType.SIGMOID;
    }

    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        this.inAct = vol;
        final Vol vol2 = vol.cloneAndZero();
        for (int i = 0; i < vol.w.length; i++) {
            vol2.w[i] = 1 / (1 + Math.exp(-vol.w[i]));
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
        return json;
    }

    @Override
    public void fromJSON(final JsonObject json) {
        this.outDepth = json.get("outDepth").getAsInt();
        this.outSX = json.get("outSX").getAsInt();
        this.outSY = json.get("outSY").getAsInt();
        this.type = Layer.LayerType.valueOf(json.get("type").getAsString());
    }

    @Override
    public double backward(final Vol output) {
        final Vol vol = this.inAct; // we need to set dw of this
        final Vol vol2 = this.outAct;
        vol.dw = Utils.zerosDouble(vol.w.length); // zero out gradient wrt data
        for (int i = 0; i < vol.w.length; i++) {
            final double v2wi = vol2.w[i];
            vol.dw[i] = v2wi * (1.0 - v2wi) * vol2.dw[i];
        }
        return 0;
    }
}
