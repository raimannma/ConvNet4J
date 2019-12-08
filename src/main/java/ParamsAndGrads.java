import java.util.ArrayList;
import java.util.List;

public class ParamsAndGrads {
    final List<Double> grads;
    private final List<Double> l1DecayList;
    private final List<Double> l2DecayList;
    double l2DecayMul, l1DecayMul;
    List<Double> params;

    public ParamsAndGrads() {
        this.params = new ArrayList<>();
        this.grads = new ArrayList<>();
        this.l1DecayList = new ArrayList<>();
        this.l2DecayList = new ArrayList<>();
        this.l1DecayMul = Double.NaN;
        this.l2DecayMul = Double.NaN;
    }
}
