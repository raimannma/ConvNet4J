package layers;

import enums.ActivationType;

public class LayerConfig {
    public int numClasses;
    public ActivationType activation;
    public Layer.LayerType type;
    public int inSX;
    public int inSY;
    public int inDepth;
    public double dropProb;
    public double biasPref;
    public int groupSize;
    public int numNeurons;
    private int filters;
    private int sx;
    private int sy;
    private int outSX;
    private int outSY;
    private int outDepth;
    private int stride;
    private int pad;
    private double l1DecayMul;
    private double l2DecayMul;
    private int depth;
    private int k;
    private int n;
    private double alpha;
    private double beta;

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

    public int getNumClasses() {
        return this.numClasses;
    }

    public void setNumClasses(final int numClasses) {
        this.numClasses = numClasses;
    }

    public ActivationType getActivation() {
        return this.activation;
    }

    public void setActivation(final ActivationType activation) {
        this.activation = activation;
    }

    public Layer.LayerType getType() {
        return this.type;
    }

    public void setType(final Layer.LayerType type) {
        this.type = type;
    }

    public int getInSX() {
        return this.inSX;
    }

    public void setInSX(final int inSX) {
        this.inSX = inSX;
    }

    public int getInSY() {
        return this.inSY;
    }

    public void setInSY(final int inSY) {
        this.inSY = inSY;
    }

    public int getInDepth() {
        return this.inDepth;
    }

    public void setInDepth(final int inDepth) {
        this.inDepth = inDepth;
    }

    double getDropProb() {
        return this.dropProb;
    }

    public void setDropProb(final double dropProb) {
        this.dropProb = dropProb;
    }

    double getBiasPref() {
        return this.biasPref;
    }

    public void setBiasPref(final double biasPref) {
        this.biasPref = biasPref;
    }

    int getGroupSize() {
        return this.groupSize;
    }

    public void setGroupSize(final int groupSize) {
        this.groupSize = groupSize;
    }

    int getNumNeurons() {
        return this.numNeurons;
    }

    public void setNumNeurons(final int numNeurons) {
        this.numNeurons = numNeurons;
    }

    int getFilters() {
        return this.filters;
    }

    public void setFilters(final int filters) {
        this.filters = filters;
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

    int getOutSX() {
        return this.outSX;
    }

    public void setOutSX(final int outSX) {
        this.outSX = outSX;
    }

    int getOutSY() {
        return this.outSY;
    }

    public void setOutSY(final int outSY) {
        this.outSY = outSY;
    }

    int getOutDepth() {
        return this.outDepth;
    }

    public void setOutDepth(final int outDepth) {
        this.outDepth = outDepth;
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

    public void setL1DecayMul(final double l1DecayMul) {
        this.l1DecayMul = l1DecayMul;
    }

    double getL2DecayMul() {
        return this.l2DecayMul;
    }

    public void setL2DecayMul(final double l2DecayMul) {
        this.l2DecayMul = l2DecayMul;
    }

    int getDepth() {
        return this.depth;
    }

    public void setDepth(final int depth) {
        this.depth = depth;
    }

    int getK() {
        return this.k;
    }

    public void setK(final int k) {
        this.k = k;
    }

    int getN() {
        return this.n;
    }

    public void setN(final int n) {
        this.n = n;
    }

    double getAlpha() {
        return this.alpha;
    }

    public void setAlpha(final double alpha) {
        this.alpha = alpha;
    }

    double getBeta() {
        return this.beta;
    }

    public void setBeta(final double beta) {
        this.beta = beta;
    }
}
