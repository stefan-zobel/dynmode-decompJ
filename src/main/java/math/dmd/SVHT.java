/*
 * Copyright 2020, 2021 Stefan Zobel
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
package math.dmd;

import net.jamu.matrix.Dimensions;
import net.jamu.matrix.MatrixD;
import net.jamu.matrix.MatrixF;

/**
 * Approximately optimal Singular Value truncation after Gavish and Donoho
 * (2014).
 * 
 * @see "https://arxiv.org/pdf/1305.5870.pdf"
 */
class SVHT {

    /** The IEEE 754 machine epsilon from Cephes: (2^-53) */
    static final double MACH_EPS_DBL = 1.11022302462515654042e-16;
    /** The IEEE 754 single precision machine epsilon: (2^-24) */
    static final float MACH_EPS_FLT = 5.9604644775390625e-8f;
    static final double TOL_DBL = 5.0 * MACH_EPS_DBL;
    static final float TOL_FLT = 5.0f * MACH_EPS_FLT;
    static final double FAIR_SHARE_DBL = 1.0 - 1e-4;
    static final float FAIR_SHARE_FLT = 1.0f - 1e-4f;

    static int thresholdD(MatrixD data, double[] singularValues) {
        if (singularValues[0] <= MACH_EPS_DBL) {
            return 0;
        }
        double omega = computeOmega(data);
        double median = medianD(singularValues);
        double cutoff = omega * median;
        return thresholdD(singularValues, cutoff);
    }

    static int thresholdF(MatrixF data, float[] singularValues) {
        if (singularValues[0] <= MACH_EPS_FLT) {
            return 0;
        }
        float omega = (float) computeOmega(data);
        float median = medianF(singularValues);
        float cutoff = omega * median;
        return thresholdF(singularValues, cutoff);
    }

    private static double medianD(double[] singularValues) {
        int len = singularValues.length;
        int endIdx = len - 1;
        for (int i = endIdx; i >= 0; --i) {
            if (singularValues[i] > TOL_DBL) {
                endIdx = i;
                break;
            }
        }
        if (endIdx < len - 1) {
            len = endIdx + 1;
        }
        if (len % 2 != 0) {
            return singularValues[(len - 1) / 2];
        } else {
            int mid = len / 2;
            return (singularValues[mid - 1] + singularValues[mid]) / 2.0;
        }
    }

    private static float medianF(float[] singularValues) {
        int len = singularValues.length;
        int endIdx = len - 1;
        for (int i = endIdx; i >= 0; --i) {
            if (singularValues[i] > TOL_FLT) {
                endIdx = i;
                break;
            }
        }
        if (endIdx < len - 1) {
            len = endIdx + 1;
        }
        if (len % 2 != 0) {
            return singularValues[(len - 1) / 2];
        } else {
            int mid = len / 2;
            return (singularValues[mid - 1] + singularValues[mid]) / 2.0f;
        }
    }

    private static double computeOmega(Dimensions data) {
        int rows = data.numRows();
        int cols = data.numColumns();
        int m = Math.min(rows, cols);
        int n = Math.max(rows, cols);
        double beta = m / (double) n;
        double betaSqr = beta * beta;
        double betaCub = betaSqr * beta;
        return 0.56 * betaCub - 0.95 * betaSqr + 1.82 * beta + 1.43;
    }

    private static int thresholdD(double[] singularValues, double cutoff) {
        int idx = 0;
        for (int i = 0; i < singularValues.length; ++i) {
            if (singularValues[i] <= cutoff) {
                // idx of last sv > cutoff
                idx = i - 1;
                break;
            }
        }
        if (idx > 0) {
            double cap = FAIR_SHARE_DBL * sumD(singularValues);
            double sum = 0.0;
            int lastIdx = 0;
            for (int i = 0; i <= idx && sum < cap; ++i) {
                sum += singularValues[i];
                lastIdx = i;
            }
            idx = Math.min(idx, lastIdx);
        }
        // estimated rank
        idx = (idx < 0) ? 0 : idx;
        return idx + 1;
    }

    private static int thresholdF(float[] singularValues, float cutoff) {
        int idx = 0;
        for (int i = 0; i < singularValues.length; ++i) {
            if (singularValues[i] <= cutoff) {
                // idx of last sv > cutoff
                idx = i - 1;
                break;
            }
        }
        if (idx > 0) {
            float cap = FAIR_SHARE_FLT * sumF(singularValues);
            float sum = 0.0f;
            int lastIdx = 0;
            for (int i = 0; i <= idx && sum < cap; ++i) {
                sum += singularValues[i];
                lastIdx = i;
            }
            idx = Math.min(idx, lastIdx);
        }
        // estimated rank
        idx = (idx < 0) ? 0 : idx;
        return idx + 1;
    }

    private static double sumD(double[] singularValues) {
        double sum = 0.0;
        for (int i = 0; i < singularValues.length; ++i) {
            double sv = singularValues[i];
            if (sv <= TOL_DBL) {
                break;
            }
            sum += sv;
        }
        return sum;
    }

    private static float sumF(float[] singularValues) {
        float sum = 0.0f;
        for (int i = 0; i < singularValues.length; ++i) {
            double sv = singularValues[i];
            if (sv <= TOL_FLT) {
                break;
            }
            sum += sv;
        }
        return sum;
    }

    private SVHT() {
        throw new AssertionError();
    }
}
