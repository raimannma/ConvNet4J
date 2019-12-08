class ParamsAndGrads {
    double[] grads;
    double l2DecayMul, l1DecayMul;
    double[] params;

    ParamsAndGrads() {
        this.params = null;
        this.grads = null;
        this.l1DecayMul = Double.NaN;
        this.l2DecayMul = Double.NaN;
    }
}
