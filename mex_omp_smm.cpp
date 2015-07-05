#include <omp.h>
#include "mex.h"
#include "matrix.h"

#define ll long long

void omp_smm(double* A, double*B, double* C, ll m, ll p, ll n, ll* irs, ll* jcs)
{
    for (ll j=0; j<p; ++j)
    {
        ll istart = jcs[j];
        ll iend = jcs[j+1];
        #pragma omp parallel for
        for (ll ii=istart; ii<iend; ++ii)
        {
            ll i = irs[ii];
            double aa = A[ii];
            for (ll k=0; k<n; ++k)
            {
                C[i+k*m] += B[j+k*p]*aa;
            }
        }
    }
}

void omp_smm2(double* A, double*B, double* C, ll m, ll p, ll n, ll* irs, ll* jcs)
{
    int nthreads, tid;
    double* cpointer[24];
    int j;

    #pragma omp parallel private(tid, j) shared(C, A, B, cpointer)
    {
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
        cpointer[tid] = new double[m*n];
        for (ll i=0; i<m*n; i++)
            cpointer[tid][i] = 0;

        //mexPrintf("%d\n", tid);

        #pragma omp for schedule(dynamic) nowait
        for (j=0; j<p; ++j)
        {
            tid = omp_get_thread_num();
            ll istart = jcs[j];
            ll iend = jcs[j+1];
            for (ll ii=istart; ii<iend; ++ii)
            {
                ll i = irs[ii]*n;
                double aa = A[ii];
                for (ll k=0; k<n; ++k)
                {
                    cpointer[tid][i+k] += B[j+k*p]*aa;
                }
            }
        }
    }
    
    //mexPrintf("%d\n", nthreads);
    

    for (int j=0; j<nthreads; j++)
    {
        for (ll i=0; i<m; i++)
        {
            for (ll k=0; k<n; k++)
                C[i+k*m] += cpointer[j][i*n+k];
        }
    }


    for (int j=0; j<nthreads; j++)
        delete[] cpointer[j];
    //delete[] cpointer;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *A, *B, *C; /* pointers to input & output matrices*/
    size_t m,n,p;      /* matrix dimensions */

    A = mxGetPr(prhs[0]); /* first sparse matrix */
    B = mxGetPr(prhs[1]); /* second full matrix */

    mwIndex * irs = mxGetIr(prhs[0]);
    mwIndex * jcs = mxGetJc(prhs[0]);

    m = mxGetM(prhs[0]);  
    p = mxGetN(prhs[0]);
    n = mxGetN(prhs[1]);

    /* create output matrix C */
    plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
    C = mxGetPr(plhs[0]);
    
    omp_smm2(A,B,C, m, p, n, (ll*)irs, (ll*)jcs);
}

