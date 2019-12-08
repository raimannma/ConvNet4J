package layers;

import com.google.gson.JsonObject;
import utils.ParamsAndGrads;
import utils.Utils;
import utils.Vol;

public class ReluLayer extends Layer {
    public ReluLayer() {
        this(new LayerConfig());
    }

    public ReluLayer(final LayerConfig opt) {
        //computed
        this.outSX = opt.getInSX();
        this.outSY = opt.getInSY();
        this.outDepth = opt.getInDepth();
        this.type = Layer.LayerType.RELU;
    }

    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        this.inAct = vol;
        final Vol vol2 = vol.clone();
        for (int i = 0; i < vol.w.length; i++) {
            if (vol2.w[i] < 0) {
                vol2.w[i] = 0;
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
        final Vol vol = this.inAct;
        final Vol vol2 = this.outAct;
        vol.dw = Utils.zerosDouble(vol.w.length);
        for (int i = 0; i < vol.w.length; i++) {
            vol.dw[i] = vol2.w[i] <= 0 ? 0 : vol2.dw[i];
        }
        return 0;
    }
}
