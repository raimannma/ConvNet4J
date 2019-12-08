class ParamsAndGrads {
    private final double[] l1DecayList;
    private final double[] l2DecayList;
    double[] grads;
    double l2DecayMul, l1DecayMul;
    double[] params;

    ParamsAndGrads() {
        this.params = null;
        this.grads = null;
        this.l1DecayList = null;
        this.l2DecayList = null;
        this.l1DecayMul = Double.NaN;
        this.l2DecayMul = Double.NaN;
    }
}
