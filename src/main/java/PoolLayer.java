import com.google.gson.JsonObject;

public class PoolLayer extends Layer {
    private final int inSY;
    private final int inSX;
    private int sx;
    private int sy;
    private int inDepth;
    private int stride;
    private int pad;
    private int[] switchX;
    private int[] switchY;

    PoolLayer() {
        this(new LayerConfig());
    }

    PoolLayer(final LayerConfig opt) {
        // required
        this.sx = opt.getSX();
        this.inDepth = opt.getInDepth();
        this.inSX = opt.getInSX();
        this.inSY = opt.getInSY();

        // optional
        this.sy = LayerConfig.getOrDefault(this.sx, opt.getSY());
        this.stride = LayerConfig.getOrDefault(2, opt.getStride());
        this.pad = LayerConfig.getOrDefault(0, opt.getPad());

        //computed
        this.outDepth = this.inDepth;
        this.outSX = (int) Math.floor((double) (this.inSX + this.pad * 2 - this.sx) / this.stride + 1);
        this.outSX = (int) Math.floor((double) (this.inSY + this.pad * 2 - this.sy) / this.stride + 1);
        this.type = LayerType.POOL;

        this.switchX = Utils.zerosInt(this.outSX * this.outSY * this.outDepth);
        this.switchY = Utils.zerosInt(this.outSX * this.outSY * this.outDepth);
    }

    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        this.inAct = vol;

        final Vol activate = new Vol(this.outSX, this.outSY, this.outDepth, 0.0);

        int n = 0; // a counter for switches
        for (int d = 0; d < this.outDepth; d++) {
            int x = -this.pad;
            int y;
            for (int ax = 0; ax < this.outSX; x += this.stride, ax++) {
                y = -this.pad;
                for (int ay = 0; ay < this.outSY; y += this.stride, ay++) {

                    // convolve centered at this particular location
                    double max = Double.MIN_VALUE;
                    int winx = -1;
                    int winy = -1;
                    for (int fx = 0; fx < this.sx; fx++) {
                        for (int fy = 0; fy < this.sy; fy++) {
                            final int oy = y + fy;
                            final int ox = x + fx;
                            if (oy >= 0 && oy < vol.sy && ox >= 0 && ox < vol.sx) {
                                final double v = vol.get(ox, oy, d);
                                // perform max pooling and store pointers to where
                                // the max came from. This will speed up backprop
                                // and can help make nice visualizations in future
                                if (v > max) {
                                    max = v;
                                    winx = ox;
                                    winy = oy;
                                }
                            }
                        }
                    }
                    this.switchX[n] = winx;
                    this.switchY[n] = winy;
                    n++;
                    activate.set(ax, ay, d, max);
                }
            }
        }
        this.outAct = activate;
        return this.outAct;
    }

    @Override
    public ParamsAndGrads[] getParamsAndGrads() {
        return new ParamsAndGrads[0];
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
        json.addProperty("pad", this.pad);
        return json;
    }

    @Override
    public void fromJSON(final JsonObject json) {
        this.outDepth = json.get("outDepth").getAsInt();
        this.outSX = json.get("outSX").getAsInt();
        this.outSY = json.get("outSY").getAsInt();
        this.type = LayerType.valueOf(json.get("type").getAsString());
        this.sx = json.get("sx").getAsInt();
        this.sy = json.get("sy").getAsInt();
        this.stride = json.get("stride").getAsInt();
        this.inDepth = json.get("inDepth").getAsInt();
        this.pad = Double.isNaN(json.get("pad").getAsInt()) ? 0 : json.get("pad").getAsInt(); // backwards compatibility
        this.switchX = Utils.zerosInt(this.outSX * this.outSY * this.outDepth); // need to re-init these appropriately
        this.switchY = Utils.zerosInt(this.outSX * this.outSY * this.outDepth);
    }

    @Override
    public double backward(final Vol output) {
        // pooling layers have no parameters, so simply compute
        // gradient wrt data here
        final Vol vol = this.inAct;
        vol.dw = Utils.zerosDouble(vol.w.length); // zero out gradient wrt data

        int n = 0;
        for (int d = 0; d < this.outDepth; d++) {
            int x = -this.pad;
            for (int ax = 0; ax < this.outSX; x += this.stride, ax++) {
                int y = -this.pad;
                for (int ay = 0; ay < this.outSY; y += this.stride, ay++) {
                    final double chainGrad = this.outAct.getGrad(ax, ay, d);
                    vol.addGrad(this.switchX[n], this.switchY[n], d, chainGrad);
                    n++;
                }
            }
        }
        return 0;
    }
}
