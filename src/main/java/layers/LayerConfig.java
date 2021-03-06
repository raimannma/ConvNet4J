package layers;

import enums.ActivationType;

import java.util.Arrays;

import static java.lang.Double.NaN;

public class LayerConfig {
    private double biasPref;
    private double dropProb;
    private Layer.LayerType type;
    private int groupSize;
    private int numNeurons;
    private ActivationType activation;
    private int numClasses;
    private int inSX;
    private int inSY;
    private int inDepth;
    private int sx;
    private int sy;
    private int depth;
    private int outSX;
    private int outSY;
    private int outDepth;
    private int filters;
    private int stride;
    private int pad;
    private double l1DecayMul;
    private double l2DecayMul;
    private int k;
    private int n;
    private double alpha;
    private double beta;

    public LayerConfig() {
        this.numClasses = -1;
        this.activation = null;
        this.type = null;
        this.inSX = -1;
        this.inSY = -1;
        this.inDepth = -1;
        this.dropProb = NaN;
        this.biasPref = NaN;
        this.groupSize = -1;
        this.numNeurons = -1;
        this.filters = -1;
        this.sx = -1;
        this.sy = -1;
        this.outSX = -1;
        this.outSY = -1;
        this.outDepth = -1;
        this.stride = -1;
        this.pad = -1;
        this.l1DecayMul = NaN;
        this.l2DecayMul = NaN;
        this.depth = -1;
        this.n = -1;
        this.alpha = NaN;
        this.beta = NaN;
    }

    static double getOrDefault(final double defaultValue, final double... values) {
        return Arrays.stream(values)
                .filter(value -> !Double.isNaN(value) && value != -1)
                .findFirst()
                .orElse(defaultValue);
    }

    static int getOrDefault(final int defaultValue, final int... values) {
        for (final int value : values) {
            if (!Double.isNaN(value) && value != -1) {
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

    public double getDropProb() {
        return this.dropProb;
    }

    public void setDropProb(final double dropProb) {
        this.dropProb = dropProb;
    }

    public double getBiasPref() {
        return this.biasPref;
    }

    public void setBiasPref(final double biasPref) {
        this.biasPref = biasPref;
    }

    public int getGroupSize() {
        return this.groupSize;
    }

    public void setGroupSize(final int groupSize) {
        this.groupSize = groupSize;
    }

    public int getNumNeurons() {
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
