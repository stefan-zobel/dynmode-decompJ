/*
 * Copyright 2025 Stefan Zobel
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package test.math.dmd;

import math.coord.LinSpace;
import math.dmd.ExactDMDV2;
import math.fun.DIndexIterator;
import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;

/**
 * Cosine example with delay = 3 and t_start <> 0.
 */
public class CosineWithDelayWorksSomehow {

    static final double x_start = 0.0;
    static final double x_end = 10.0;
    static final int x_num = 1;

    static final double t_start = 1.0;
    static final double t_end = t_start + 4.0 * Math.PI;
    static final int t_num = 600;

    // need that many delays to correctly determine the rank
    static final int delays = 3;

    // space dimension
    static final LinSpace xi = LinSpace.linspace(x_start, x_end, x_num);
    // time dimension
    static final LinSpace ti = LinSpace.linspace(t_start, t_end, t_num);

    public static void main(String[] args) {
        // build data 'measurements' matrix
        MatrixD data = setupMeasurementsMatrix(ti, delays);

        // step size
        double deltaT = (t_end - t_start) / (t_num - 1);
        System.out.println("deltaT: " + deltaT);

        ExactDMDV2 dmd = new ExactDMDV2(data, deltaT, t_start).compute();
        System.out.println("Estimated rank: " + dmd.getRank());

        // predict the same interval that was used to compute the DMD
        MatrixD pred = dmd.predict(t_start, t_num - delays);

        // XXX (would be necessary for Math.sin() to pass)
//        pred.setUnsafe(0, 0, data.getUnsafe(0, 0));
//        pred.setUnsafe(pred.endRow(), pred.endCol(), data.getUnsafe(data.endRow(), data.endCol()));
        // XXX
        System.out.println("reconstructed:" + pred);
        System.out.println("original     :" + data);

        // compare Frobenius norms for approximate equality
        double normDmd = pred.normF();
        double normData = data.normF();
        System.out.println("reconstructed: " + normDmd);
        System.out.println("original     : " + normData);
        double largerNorm = Math.max(normDmd, normData);
        double smallerNorm = Math.min(normDmd, normData);
        System.out.println("ratio: " + largerNorm / smallerNorm);
        MatrixD relErr = RelativeError.compute(data, pred);
        System.out.println(relErr);
        System.out.println(RelativeError.avgRelErrorOverall(relErr));
        System.out.println("Matrices.approxEqual: " + Matrices.approxEqual(data, pred, 1.0e-3));
        System.out.println("Matrices.distance: " + Matrices.distance(data, pred));

        // now attempt to predict the future starting from 4.0 * PI for t_num
        // predictions with the same stepsize
        int t_num = 100;
        double t_start = t_end;
        double t_end = t_start + dmd.getDeltaT() * (t_num - 1);
        deltaT = (t_end - t_start) / (t_num - 1);
        System.out.println("\ndeltaT: " + deltaT);

        // setup a LinSpace corresponding to the prediction interval
        LinSpace newTime = LinSpace.linspace(t_start, t_end, t_num);
        // and create the data for that interval
        MatrixD newData = setupMeasurementsMatrix(newTime, delays);

        MatrixD fut = dmd.predict(t_start, t_num - delays);
        // XXX (would be necessary for Math.sin() to pass)
//        fut.setUnsafe(0, 0, newData.getUnsafe(0, 0));
//        fut.setUnsafe(fut.endRow(), fut.endCol(), newData.getUnsafe(newData.endRow(), newData.endCol()));
        // XXX
        System.out.println("future       :" + fut);

        System.out.println("new data     :" + newData);
        // and compare the Frobenius norms
        double normPred = fut.normF();
        double normRea = newData.normF();
        System.out.println("predicted    : " + normPred);
        System.out.println("realized     : " + normRea);
        largerNorm = Math.max(normPred, normRea);
        smallerNorm = Math.min(normPred, normRea);
        System.out.println("ratio: " + largerNorm / smallerNorm);
        relErr = RelativeError.compute(newData, fut);
        System.out.println(relErr);
        System.out.println(RelativeError.avgRelErrorOverall(relErr));
        System.out.println("Matrices.approxEqual: " + Matrices.approxEqual(newData, fut, 1.0e-3));
        System.out.println("Matrices.distance: " + Matrices.distance(newData, fut));
    }

    // setup Hankel matrix
    private static MatrixD timeDelayed(MatrixD m, int delays) {
        if (delays < 0 || delays >= m.numColumns()) {
            throw new IllegalArgumentException("delays: " + delays);
        }
        if (delays == 0) {
            return m; // copy ?
        }
        MatrixD H = Matrices.createD(m.numRows() * (delays + 1), m.numColumns() - delays);
        for (int col = 0; col < m.numColumns() - delays; ++col) {
            int row_ = 0;
            int col_ = col;
            for (int row = 0; row < m.numRows() * (delays + 1); ++row) {
                H.setUnsafe(row, col, m.getUnsafe(row_, col_));
                ++row_;
                if (row_ == m.numRows()) {
                    row_ = 0;
                    ++col_;
                }
            }
        }
        return H;
    }

    private static MatrixD setupMeasurementsMatrix(LinSpace time, int delays) {
        return timeDelayed(setupMeasurementsMatrix_(time), delays);
    }

    private static MatrixD setupMeasurementsMatrix_(LinSpace time) {
        // build data 'measurements' matrix
        MatrixD X_ = Matrices.createD(xi.size(), time.size());

        for (DIndexIterator tIt = time.iterator(); tIt.hasNext(); /**/) {
            int colIdx = tIt.nextIndex() - 1;
            double t = tIt.next();
            for (DIndexIterator xIt = xi.iterator(); xIt.hasNext(); /**/) {
                int rowIdx = xIt.nextIndex() - 1;
                xIt.next();
                X_.set(rowIdx, colIdx, Math.cos(t));
            }
        }

        return X_;
    }
}
