package learning;

import enums.TrainerMethod;

class TrainerOptions {
    double learningRate;
    double l2Decay;
    double batchSize;
    double momentum;
    private double l1Decay;
    private TrainerMethod method;
    private double ro;
    private double epsilon;
    private double beta1;
    private double beta2;

    double getLearningRate() {
        return this.learningRate;
    }

    public void setLearningRate(final double learningRate) {
        this.learningRate = learningRate;
    }

    double getL1Decay() {
        return this.l1Decay;
    }

    public void setL1Decay(final double l1Decay) {
        this.l1Decay = l1Decay;
    }

    double getL2Decay() {
        return this.l2Decay;
    }

    public void setL2Decay(final double l2Decay) {
        this.l2Decay = l2Decay;
    }

    double getBatchSize() {
        return this.batchSize;
    }

    public void setBatchSize(final double batchSize) {
        this.batchSize = batchSize;
    }

    TrainerMethod getMethod() {
        return this.method;
    }

    public void setMethod(final TrainerMethod method) {
        this.method = method;
    }

    double getMomentum() {
        return this.momentum;
    }

    public void setMomentum(final double momentum) {
        this.momentum = momentum;
    }

    double getRo() {
        return this.ro;
    }

    public void setRo(final double ro) {
        this.ro = ro;
    }

    double getEpsilon() {
        return this.epsilon;
    }

    public void setEpsilon(final double epsilon) {
        this.epsilon = epsilon;
    }

    double getBeta1() {
        return this.beta1;
    }

    public void setBeta1(final double beta1) {
        this.beta1 = beta1;
    }

    double getBeta2() {
        return this.beta2;
    }

    public void setBeta2(final double beta2) {
        this.beta2 = beta2;
    }
}
