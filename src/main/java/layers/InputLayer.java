package layers;

import com.google.gson.JsonObject;
import utils.ParamsAndGrads;
import utils.Vol;

public class InputLayer extends Layer {

    public InputLayer() {
        this(new LayerConfig());
    }

    public InputLayer(final LayerConfig opt) {
        // required
        this.outDepth = LayerConfig.getOrDefault(0, opt.getOutDepth(), opt.getInDepth(), opt.getDepth());

        // optional
        this.outSX = LayerConfig.getOrDefault(1, opt.getOutSX(), opt.getSX());
        this.outSY = LayerConfig.getOrDefault(1, opt.getOutSY(), opt.getSY());

        // computed
        this.type = LayerType.INPUT;
    }

    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        this.inAct = vol;
        this.outAct = vol;
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
        this.type = LayerType.valueOf(json.get("type").toString());
    }

    @Override
    public double backward(final Vol output) {
        return 0;
    }
}
