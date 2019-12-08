package layers;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import utils.ParamsAndGrads;
import utils.Utils;
import utils.Vol;

import java.util.ArrayList;
import java.util.List;

public class FullyConnectedLayer extends Layer {
    private Vol biases;
    private ArrayList<Vol> filters;
    private double l1DecayMul;
    private double l2DecayMul;
    private int numInputs;

    public FullyConnectedLayer() {
        this(new LayerConfig());
    }

    public FullyConnectedLayer(final LayerConfig opt) {
        // required
        this.outDepth = LayerConfig.getOrDefault(opt.getNumNeurons(), opt.getFilters());


        // optional
        this.l1DecayMul = LayerConfig.getOrDefault(0, opt.getL1DecayMul());
        this.l2DecayMul = LayerConfig.getOrDefault(1, opt.getL2DecayMul());

        //computed
        this.numInputs = opt.getInSX() * opt.getInSY() * opt.getInDepth();
        this.outSX = 1;
        this.outSY = 1;
        this.type = LayerType.FC;

        //init
        final double bias = LayerConfig.getOrDefault(0, opt.getBiasPref());
        this.filters = new ArrayList<>();
        for (int i = 0; i < this.outDepth; i++) {
            this.filters.add(new Vol(1, 1, this.numInputs));
        }
        this.biases = new Vol(1, 1, this.outDepth, bias);
    }

    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        this.inAct = vol;
        final Vol A = new Vol(1, 1, this.outDepth, 0.0);
        final double[] Vw = vol.w;
        for (int i = 0; i < this.outDepth; i++) {
            double a = 0;
            final double[] wi = this.filters.get(i).w;
            for (int j = 0; j < this.numInputs; j++) {
                a += Vw[j] * wi[j]; // for efficiency use Vols directly for now
            }
            a += this.biases.w[i];
            A.w[i] = a;
        }
        this.outAct = A;
        return this.outAct;
    }

    @Override
    public ParamsAndGrads[] getParamsAndGrads() {
        final List<ParamsAndGrads> out = new ArrayList<>();
        for (int i = 0; i < this.outDepth; i++) {
            final ParamsAndGrads pg = new ParamsAndGrads();
            pg.params = this.filters.get(i).w;
            pg.grads = this.filters.get(i).dw;
            pg.l1DecayMul = this.l1DecayMul;
            pg.l2DecayMul = this.l2DecayMul;
            out.add(pg);
        }

        final ParamsAndGrads pg = new ParamsAndGrads();
        pg.params = this.biases.w;
        pg.grads = this.biases.dw;
        pg.l1DecayMul = 0;
        pg.l2DecayMul = 0;
        out.add(pg);

        return out.toArray(ParamsAndGrads[]::new);
    }

    @Override
    public JsonObject toJSON() {
        final JsonObject json = new JsonObject();
        json.addProperty("outDepth", this.outDepth);
        json.addProperty("outSX", this.outSX);
        json.addProperty("outSY", this.outSY);
        json.addProperty("type", this.type.toString());
        json.addProperty("numInputs", this.numInputs);
        json.addProperty("l1DecayMul", this.l1DecayMul);
        json.addProperty("l2DecayMul", this.l2DecayMul);

        final JsonArray jsonFilters = new JsonArray();
        for (final Vol filter : this.filters) {
            jsonFilters.add(filter.toJSON());
        }
        json.add("filters", jsonFilters);
        json.add("biases", this.biases.toJSON());
        return json;
    }

    @Override
    public void fromJSON(final JsonObject json) {
        this.outDepth = json.get("outDepth").getAsInt();
        this.outSX = json.get("outSX").getAsInt();
        this.outSY = json.get("outSY").getAsInt();
        this.type = LayerType.valueOf(json.get("type").getAsString());
        this.numInputs = json.get("numInputs").getAsInt();
        this.l1DecayMul = Double.isNaN(json.get("l1DecayMul").getAsDouble()) ? 1 : json.get("l1DecayMul").getAsDouble();
        this.l2DecayMul = Double.isNaN(json.get("l2DecayMul").getAsDouble()) ? 1 : json.get("l2DecayMul").getAsDouble();
        this.filters = new ArrayList<>();
        final JsonArray jsonFilters = json.get("filters").getAsJsonArray();
        for (int i = 0; i < jsonFilters.size(); i++) {
            this.filters.add(Vol.fromJSON(jsonFilters.get(i).getAsJsonObject()));
        }
        this.biases = Vol.fromJSON(json.get("biases").getAsJsonObject());
    }

    @Override
    public double backward(final Vol output) {
        final Vol V = this.inAct;
        V.dw = Utils.zerosDouble(V.w.length); // zero out the gradient in input utils.Vol

        // compute gradient wrt weights and data
        for (int i = 0; i < this.outDepth; i++) {
            final Vol tfi = this.filters.get(i);
            final double chainGrad = this.outAct.dw[i];
            for (int j = 0; j < this.numInputs; j++) {
                V.dw[j] += tfi.w[j] * chainGrad; // grad wrt input data
                tfi.dw[j] += V.w[j] * chainGrad; // grad wrt params
            }
            this.biases.dw[i] += chainGrad;
        }
        return 0;
    }
}
