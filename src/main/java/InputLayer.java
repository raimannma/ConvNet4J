import com.google.gson.JsonObject;

public class InputLayer extends Layer {
    private Vol in_act;

    InputLayer() {
        this(new LayerConfig());
    }

    InputLayer(final LayerConfig opt) {
        // required
        this.out_depth = LayerConfig.getOrDefault(0, opt.getOutDepth(), opt.getInDepth(), opt.getDepth());

        // optional
        this.out_sx = LayerConfig.getOrDefault(1, opt.getOutSX(), opt.getSX());
        this.out_sy = LayerConfig.getOrDefault(1, opt.getOutSY(), opt.getSY());

        // computed
        this.type = LayerType.INPUT;
    }

    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        this.in_act = vol;
        this.out_act = vol;
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
        return json;
    }

    @Override
    public void fromJSON(final JsonObject json) {
        this.out_depth = json.get("out_depth").getAsInt();
        this.out_sx = json.get("out_sx").getAsInt();
        this.out_sy = json.get("out_sy").getAsInt();
        this.type = LayerType.valueOf(json.get("type").toString());
    }

    @Override
    public double backward(final Vol output) {
        return 0;
    }
}
