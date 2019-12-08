package learning;

import enums.TrainerMethod;
import utils.ParamsAndGrads;
import utils.Utils;
import utils.Vol;

import java.util.*;

class Trainer {

    private final Network net;
    private final double learningRate;
    private final double l1Decay;
    private final double l2Decay;
    private final double batchSize;
    private final double momentum;
    private final double ro;
    private final double epsilon;
    private final double beta1;
    private final double beta2;
    private final List<List<Double>> gSum;
    private final List<List<Double>> xSum;
    private final TrainerMethod method;
    private double k;

    Trainer(final Network network, final TrainerOptions options) {
        this.net = network;
        this.learningRate = options.getLearningRate();
        this.l1Decay = options.getL1Decay();
        this.l2Decay = options.getL2Decay();
        this.batchSize = options.getBatchSize();
        this.method = options.getMethod();

        this.momentum = options.getMomentum();
        this.ro = options.getRo();
        this.epsilon = options.getEpsilon();
        this.beta1 = options.getBeta1();
        this.beta2 = options.getBeta2();

        this.k = 0;
        this.gSum = new ArrayList<>();
        this.xSum = new ArrayList<>();

    }

    Map<String, Double> train(final Vol x, final Vol y) {
        this.net.forward(x, true);

        final double costLoss = this.net.backward(y);
        double l2DecayLoss = 0;
        double l1DecayLoss = 0;

        this.k++;
        if (this.k % this.batchSize == 0) {
            final ParamsAndGrads[] pgList = this.net.getParamsAndGrads();

            if (this.gSum.size() == 0 && (this.method != TrainerMethod.SGD || this.momentum > 0)) {
                for (final ParamsAndGrads paramsAndGrads : pgList) {
                    this.gSum.add(Utils.zerosDoubleList(paramsAndGrads.params.length));
                    if (this.method == TrainerMethod.ADAM || this.method == TrainerMethod.ADADELTA) {
                        this.xSum.add(Utils.zerosDoubleList(paramsAndGrads.params.length));
                    } else {
                        this.xSum.add(Collections.emptyList());
                    }
                }
            }

            for (int i = 0; i < pgList.length; i++) {
                final ParamsAndGrads pg = pgList[i];
                final double[] p = pg.params;
                final double[] g = pg.grads;

                final double l1DecayMul = Double.isNaN(pg.l1DecayMul) ? 1 : pg.l1DecayMul;
                final double l2DecayMul = Double.isNaN(pg.l2DecayMul) ? 1 : pg.l2DecayMul;
                final double l1Decay = this.l1Decay * l1DecayMul;
                final double l2Decay = this.l2Decay * l2DecayMul;

                for (int j = 0; j < p.length; j++) {
                    l2DecayLoss += l2Decay * p[j] * p[j] / 2; // accumulate weight decay loss
                    l1DecayLoss += l1Decay * Math.abs(p[j]);
                    final double l1grad = l1Decay * (p[j] > 0 ? 1 : -1);
                    final double l2grad = l2Decay * (p[j]);

                    final double gij = (l2grad + l1grad + g[j]) / this.batchSize; // raw batch gradient

                    final List<Double> gsumi = this.gSum.get(i);
                    final List<Double> xsumi = this.xSum.get(i);
                    if (this.method == TrainerMethod.ADAM) {
                        // adam update
                        gsumi.set(j, gsumi.get(j) * this.beta1 + (1 - this.beta1) * gij); // update biased first moment estimate
                        xsumi.set(j, xsumi.get(j) * this.beta2 + (1 - this.beta2) * gij * gij); // update biased second moment estimate
                        final double biasCorr1 = gsumi.get(j) * (1 - Math.pow(this.beta1, this.k)); // correct bias first moment estimate
                        final double biasCorr2 = xsumi.get(j) * (1 - Math.pow(this.beta2, this.k)); // correct bias second moment estimate
                        final double dx = -this.learningRate * biasCorr1 / (Math.sqrt(biasCorr2) + this.epsilon);
                        p[j] += dx;
                    } else if (this.method == TrainerMethod.ADAGRAD) {
                        // adagrad update
                        gsumi.set(j, gsumi.get(j) + gij * gij);
                        final double dx = -this.learningRate / Math.sqrt(gsumi.get(j) + this.epsilon) * gij;
                        p[j] += dx;
                    } else if (this.method == TrainerMethod.WINDOWGRAD) {
                        // this is adagrad but with a moving window weighted average
                        // so the gradient is not accumulated over the entire history of the run.
                        // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
                        gsumi.set(j, this.ro * gsumi.get(j) + (1 - this.ro) * gij * gij);
                        final double dx = -this.learningRate / Math.sqrt(gsumi.get(j) + this.epsilon) * gij; // eps added for better conditioning
                        p[j] += dx;
                    } else if (this.method == TrainerMethod.ADADELTA) {
                        gsumi.set(j, this.ro * gsumi.get(j) + (1 - this.ro) * gij * gij);
                        final double dx = -Math.sqrt((xsumi.get(j) + this.epsilon) / (gsumi.get(j) + this.epsilon)) * gij;
                        xsumi.set(j, this.ro * xsumi.get(j) + (1 - this.ro) * dx * dx); // yes, xsum lags behind gsum by 1.
                        p[j] += dx;
                    } else if (this.method == TrainerMethod.NESTEROV) {
                        double dx = gsumi.get(j);
                        gsumi.set(j, gsumi.get(j) * this.momentum + this.learningRate * gij);
                        dx = this.momentum * dx - (1.0 + this.momentum) * gsumi.get(j);
                        p[j] += dx;
                    } else {
                        // assume SGD
                        if (this.momentum > 0.0) {
                            // momentum update
                            final double dx = this.momentum * gsumi.get(j) - this.learningRate * gij; // step
                            gsumi.set(j, dx); // back this up for next iteration of momentum
                            p[j] += dx;
                        } else {
                            // vanilla sgd
                            p[j] += -this.learningRate * gij;
                        }
                    }
                    g[j] = 0; // zero out gradient so that we can begin accumulating anew
                }
            }
        }
        final Map<String, Double> output = new HashMap<>();
        output.put("l1DecayLoss", l1DecayLoss);
        output.put("l2DecayLoss", l2DecayLoss);
        output.put("costLoss", costLoss);
        output.put("loss", costLoss + l1DecayLoss + l2DecayLoss);
        return output;
    }
}
