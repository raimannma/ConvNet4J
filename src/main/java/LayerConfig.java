class LayerConfig {
    Layer.LayerType type;
    int in_sx;
    int in_sy;
    int in_depth;
    private int filters;
    private double biasPref;
    private int sx;
    private int sy;
    private int out_sx;
    private int out_sy;
    private int out_depth;
    private int stride;
    private int pad;
    private double l1_decay_mul;
    private double l2_decay_mul;
    private int numNeurons;
    private double dropProb;

    static double getOrDefault(final double value, final double defaultValue) {
        return Double.isNaN(value) ? defaultValue : value;
    }

    static int getOrDefault(final int value, final int defaultValue) {
        return Double.isNaN(value) ? defaultValue : value;
    }

    int getFilters() {
        return this.filters;
    }

    public void setFilters(final int filters) {
        this.filters = filters;
    }

    double getBiasPref() {
        return this.biasPref;
    }

    public void setBiasPref(final double biasPref) {
        this.biasPref = biasPref;
    }

    public Layer.LayerType getType() {
        return this.type;
    }

    void setType(final Layer.LayerType type) {
        this.type = type;
    }

    int getSX() {
        return this.sx;
    }

    public void setSX(final int sx) {
        this.sx = sx;
    }

    int getSY() {
        return this.sy;
    }

    public void setSY(final int sy) {
        this.sy = sy;
    }

    int getInSX() {
        return this.in_sx;
    }

    public void setInSX(final int in_sx) {
        this.in_sx = in_sx;
    }

    int getInSY() {
        return this.in_sy;
    }

    public void setInSY(final int in_sy) {
        this.in_sy = in_sy;
    }

    int getInDepth() {
        return this.in_depth;
    }

    public void setInDepth(final int in_depth) {
        this.in_depth = in_depth;
    }

    int getOutSX() {
        return this.out_sx;
    }

    public void setOutSX(final int out_sx) {
        this.out_sx = out_sx;
    }

    int getOutSY() {
        return this.out_sy;
    }

    public void setOutSY(final int out_sy) {
        this.out_sy = out_sy;
    }

    int getOutDepth() {
        return this.out_depth;
    }

    public void setOutDepth(final int out_depth) {
        this.out_depth = out_depth;
    }

    int getStride() {
        return this.stride;
    }

    public void setStride(final int stride) {
        this.stride = stride;
    }

    int getPad() {
        return this.pad;
    }

    public void setPad(final int pad) {
        this.pad = pad;
    }

    double getL1DecayMul() {
        return this.l1_decay_mul;
    }

    public void setL1DecayMul(final double l1_decay_mul) {
        this.l1_decay_mul = l1_decay_mul;
    }

    double getL2DecayMul() {
        return this.l2_decay_mul;
    }

    public void setL2DecayMul(final double l2_decay_mul) {
        this.l2_decay_mul = l2_decay_mul;
    }

    int getNumNeurons() {
        return this.numNeurons;
    }

    double getDropProb() {
        return this.dropProb;
    }
}
