import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.util.ArrayList;
import java.util.List;

class ConvolutionalLayer extends Layer {
    private final int inSX;
    private final int inSY;
    private Vol biases;
    private int pad;
    private double l1DecayMul;
    private double l2DecayMul;
    private ArrayList<Vol> filters;
    private int sx;
    private int sy;
    private int inDepth;
    private int stride;


    ConvolutionalLayer() {
        this(new LayerConfig());
    }

    ConvolutionalLayer(final LayerConfig opt) {
        //required
        this.outDepth = opt.getFilters();
        this.sx = opt.getSX();
        this.inDepth = opt.getInDepth();
        this.inSX = opt.getInSX();
        this.inSY = opt.getInSY();

        //optional
        this.sy = LayerConfig.getOrDefault(opt.getSY(), this.sx);
        this.stride = LayerConfig.getOrDefault(opt.getStride(), 1); // stride at which we apply filters to input volume
        this.pad = LayerConfig.getOrDefault(opt.getPad(), 0); // amount of 0 padding to add around borders of input volume
        this.l1DecayMul = LayerConfig.getOrDefault(0, opt.getL1DecayMul());
        this.l2DecayMul = LayerConfig.getOrDefault(1, opt.getL2DecayMul());

        // computed
        // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
        // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
        // final application.
        this.outSX = (int) Math.floor((double) (this.inSX + this.pad * 2 - this.sx) / this.stride + 1);
        this.outSY = (int) Math.floor((double) (this.inSY + this.pad * 2 - this.sy) / this.stride + 1);
        this.type = LayerType.CONVOLUTIONAL;

        final double bias = LayerConfig.getOrDefault(0, opt.getBiasPref());
        this.filters = new ArrayList<>();
        for (int i = 0; i < this.outDepth; i++) {
            this.filters.add(new Vol(this.sx, this.sy, this.inDepth));
        }
        this.biases = new Vol(1, 1, this.outDepth, bias);
    }


    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        this.inAct = vol;
        final Vol A = new Vol(
                Double.isNaN(this.outSX) ? 0 : this.outSX,
                Double.isNaN(this.outSY) ? 0 : this.outSY,
                Double.isNaN(this.outDepth) ? 0 : this.outDepth,
                0);

        final int volSX = Double.isNaN(vol.sx) ? 0 : vol.sx;
        final int volSY = Double.isNaN(vol.sy) ? 0 : vol.sy;
        final double xyStride = Double.isNaN(this.stride) ? 0 : this.stride;

        for (int d = 0; d < this.outDepth; d++) {
            final Vol f = this.filters.get(d);
            int y = Double.isNaN(this.pad) ? 0 : -this.pad;
            for (int ay = 0; ay < this.outSY; y += xyStride, ay++) {  // xyStride
                int x = Double.isNaN(this.pad) ? 0 : -this.pad;
                for (int ax = 0; ax < this.outSX; x += xyStride, ax++) {  // xyStride
                    // convolve centered at this particular location
                    double a = 0.0;
                    for (int fy = 0; fy < f.sy; fy++) {
                        final int oy = y + fy; // coordinates in the original input array coordinates
                        for (int fx = 0; fx < f.sx; fx++) {
                            final int ox = x + fx;
                            if (oy >= 0 && oy < volSY && ox >= 0 && ox < volSX) {
                                for (int fd = 0; fd < f.depth; fd++) {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    a += f.w[((f.sx * fy) + fx) * f.depth + fd] * vol.w[((volSX * oy) + ox) * vol.depth + fd];
                                }
                            }
                        }
                    }
                    a += this.biases.w[d];
                    A.set(ax, ay, d, a);
                }
            }
        }
        this.outAct = A;
        return A;
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
        final ParamsAndGrads paramsAndGrads = new ParamsAndGrads();
        paramsAndGrads.params = this.biases.w;
        paramsAndGrads.grads = this.biases.dw;
        paramsAndGrads.l1DecayMul = 0;
        paramsAndGrads.l2DecayMul = 0;
        out.add(paramsAndGrads);

        return out.toArray(ParamsAndGrads[]::new);
    }

    @Override
    public JsonObject toJSON() {
        final JsonObject json = new JsonObject();
        json.addProperty("sx", this.sx);
        json.addProperty("sy", this.sy);
        json.addProperty("stride", this.stride);
        json.addProperty("inDepth", this.inDepth);
        json.addProperty("outDepth", this.outDepth);
        json.addProperty("outSX", this.outSX);
        json.addProperty("outSY", this.outSY);
        json.addProperty("type", this.type.toString());
        json.addProperty("l1DecayMul", this.l1DecayMul);
        json.addProperty("l2DecayMul", this.l2DecayMul);
        json.addProperty("pad", this.pad);

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
        this.sx = json.get("sx").getAsInt(); // filter size in x, y dims
        this.sy = json.get("sy").getAsInt();
        this.stride = json.get("stride").getAsInt();
        this.inDepth = json.get("inDepth").getAsInt(); // depth of input volume
        this.filters = new ArrayList<>();
        this.l1DecayMul = Double.isNaN(json.get("l1DecayMul").getAsDouble()) ? 1 : json.get("l1DecayMul").getAsDouble();
        this.l2DecayMul = Double.isNaN(json.get("l2DecayMul").getAsDouble()) ? 1 : json.get("l2DecayMul").getAsDouble();
        this.pad = Double.isNaN(json.get("pad").getAsDouble()) ? 0 : json.get("pad").getAsInt();
        final JsonArray jsonFilters = json.get("filters").getAsJsonArray();
        for (int i = 0; i < jsonFilters.size(); i++) {
            this.filters.add(Vol.fromJSON(jsonFilters.get(i).getAsJsonObject()));
        }
        this.biases = Vol.fromJSON(json.get("biases").getAsJsonObject());
    }

    @Override
    public double backward(final Vol output) {
        final Vol V = this.inAct;
        V.dw = Utils.zerosDouble(V.w.length); // zero out gradient wrt bottom data, we're about to fill it

        final int volSX = Double.isNaN(V.sx) ? 0 : V.sx;
        final int volSY = Double.isNaN(V.sy) ? 0 : V.sy;
        final int xyStride = Double.isNaN(this.stride) ? 0 : this.stride;

        for (int d = 0; d < this.outDepth; d++) {
            final Vol f = this.filters.get(d);
            int y = Double.isNaN(this.pad) ? 0 : -this.pad;
            for (int ay = 0; ay < this.outSY; y += xyStride, ay++) {  // xyStride
                int x = Double.isNaN(this.pad) ? 0 : -this.pad;
                for (int ax = 0; ax < this.outSX; x += xyStride, ax++) {  // xyStride
                    // convolve centered at this particular location
                    final double chainGrad = this.outAct.getGrad(ax, ay, d); // gradient from above, from chain rule
                    for (int fy = 0; fy < f.sy; fy++) {
                        final int oy = y + fy; // coordinates in the original input array coordinates
                        for (int fx = 0; fx < f.sx; fx++) {
                            final int ox = x + fx;
                            if (oy >= 0 && oy < volSY && ox >= 0 && ox < volSX) {
                                for (int fd = 0; fd < f.depth; fd++) {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    final int ix1 = ((volSX * oy) + ox) * V.depth + fd;
                                    final int ix2 = ((f.sx * fy) + fx) * f.depth + fd;
                                    f.dw[ix2] += V.w[ix1] * chainGrad;
                                    V.dw[ix1] += f.w[ix2] * chainGrad;
                                }
                            }
                        }
                    }
                    this.biases.dw[d] += chainGrad;
                }
            }
        }
        return 0;
    }
}