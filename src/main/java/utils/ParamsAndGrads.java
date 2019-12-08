package utils;

public class ParamsAndGrads {
    public double[] grads;
    public double l2DecayMul;
    public double l1DecayMul;
    public double[] params;

    public ParamsAndGrads() {
        this.params = null;
        this.grads = null;
        this.l1DecayMul = Double.NaN;
        this.l2DecayMul = Double.NaN;
    }
}
