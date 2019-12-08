import com.google.gson.JsonObject;

public class SoftmaxLayer extends Layer {
    private int numInputs;
    private double[] es;

    SoftmaxLayer() {
        this(new LayerConfig());
    }

    SoftmaxLayer(final LayerConfig opt) {
        //computed
        this.numInputs = opt.getInSX() * opt.getInSY() * opt.getInDepth();
        this.outDepth = this.numInputs;
        this.outSX = 1;
        this.outSY = 1;
        this.type = LayerType.SOFTMAX;
    }

    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        this.inAct = vol;

        final Vol outActivation = new Vol(1, 1, this.outDepth, 0);

        final double maxActivation = Utils.max(vol.w);

        final double[] es = new double[this.outDepth];
        double eSum = 0;
        for (int i = 0; i < es.length; i++) {
            final double e = Math.exp(vol.w[i] - maxActivation);
            eSum += e;
            es[i] = e;
        }
        for (int i = 0; i < es.length; i++) {
            es[i] /= eSum;
            outActivation.w[i] = es[i];
        }

        this.es = es;
        this.outAct = outActivation;
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
        json.addProperty("numInputs", this.numInputs);
        return json;
    }

    @Override
    public void fromJSON(final JsonObject json) {
        this.outDepth = json.get("outDepth").getAsInt();
        this.outSX = json.get("outSX").getAsInt();
        this.outSY = json.get("outSY").getAsInt();
        this.type = LayerType.valueOf(json.get("type").getAsString());
        this.numInputs = json.get("numInputs").getAsInt();
    }

    @Override
    public double backward(final Vol output) {
        final Vol x = this.inAct;
        x.dw = Utils.zerosDouble(x.w.length);
        for (int i = 0; i < this.outDepth; i++) {
            final int indicator = i == (int) output.get(0, 0, 0) ? 1 : 0;
            x.dw[i] = this.es[i] - indicator;
        }
        return Math.log(this.es[(int) output.get(0, 0, 0)]);
    }
}
