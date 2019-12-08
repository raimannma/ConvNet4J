import com.google.gson.JsonObject;

public class TanhLayer extends Layer {
    TanhLayer() {
        this(new LayerConfig());
    }

    TanhLayer(final LayerConfig opt) {
        //computed
        this.outSX = opt.getInSX();
        this.outSY = opt.getInSY();
        this.outDepth = opt.getInDepth();
        this.type = LayerType.TANH;
    }

    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        this.inAct = vol;
        final Vol vol2 = vol.cloneAndZero();
        for (int i = 0; i < vol.w.length; i++) {
            vol2.w[i] = Utils.tanh(vol.w[i]);
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
        this.type = LayerType.valueOf(json.get("type").getAsString());
    }

    @Override
    public double backward(final Vol output) {
        final Vol vol = this.inAct; // we need to set dw of this
        final Vol vol2 = this.outAct;
        vol.dw = Utils.zerosDouble(vol.w.length); // zero out gradient wrt data
        for (int i = 0; i < vol.w.length; i++) {
            final double v2wi = vol2.w[i];
            vol.dw[i] = (1.0 - v2wi * v2wi) * vol2.dw[i];
        }
        return 0;
    }
}
