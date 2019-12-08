class LayerConfig {
    Activation.ActivationType activation;
    Layer.LayerType type;
    int inSX;
    int inSY;
    int inDepth;
    double dropProb;
    double biasPref;
    int groupSize;
    private int filters;
    private int sx;
    private int sy;
    private int outSX;
    private int outSY;
    private int out_depth;
    private int stride;
    private int pad;
    private double l1DecayMul;
    private double l2DecayMul;
    private int numNeurons;
    private int depth;

    static double getOrDefault(final double defaultValue, final double... values) {
        for (final double value : values) {
            if (!Double.isNaN(value)) {
                return value;
            }
        }
        return defaultValue;
    }

    static int getOrDefault(final int defaultValue, final int... values) {
        for (final int value : values) {
            if (!Double.isNaN(value)) {
                return value;
            }
        }
        return defaultValue;
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
        return this.inSX;
    }

    public void setInSX(final int in_sx) {
        this.inSX = in_sx;
    }

    int getInSY() {
        return this.inSY;
    }

    public void setInSY(final int in_sy) {
        this.inSY = in_sy;
    }

    int getInDepth() {
        return this.inDepth;
    }

    public void setInDepth(final int in_depth) {
        this.inDepth = in_depth;
    }

    int getOutSX() {
        return this.outSX;
    }

    public void setOutSX(final int out_sx) {
        this.outSX = out_sx;
    }

    int getOutSY() {
        return this.outSY;
    }

    public void setOutSY(final int out_sy) {
        this.outSY = out_sy;
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
        return this.l1DecayMul;
    }

    public void setL1DecayMul(final double l1_decay_mul) {
        this.l1DecayMul = l1_decay_mul;
    }

    double getL2DecayMul() {
        return this.l2DecayMul;
    }

    public void setL2DecayMul(final double l2_decay_mul) {
        this.l2DecayMul = l2_decay_mul;
    }

    int getNumNeurons() {
        return this.numNeurons;
    }

    double getDropProb() {
        return this.dropProb;
    }

    void setDropProb(final double dropProb) {
        this.dropProb = dropProb;
    }

    int getDepth() {
        return this.depth;
    }

    int getGroupSize() {
        return this.groupSize;
    }

    void setGroupSize(final int groupSize) {
        this.groupSize = groupSize;
    }
}
