/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Sasikanth Avancha (Intel Corp.)
******************************************************************************/

#if defined(LIBXSMM_DNN_POOLING_FWD_BF16)
# define _mm512_load_act(A)     _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(A))),16))
#if 1
# define _mm512_roundbf16rne(A) LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(A)
# define _mm512_stream_act(A,B) _mm256_stream_si256((__m256i*)(A),_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_roundbf16rne((B)),16)))
# define _mm512_store_act(A,B)  _mm256_storeu_si256((__m256i*)(A),_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_roundbf16rne((B)),16)))
#else
# define _mm512_stream_act(A,B) _mm256_stream_si256((__m256i*)(A),_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_castps_si512((B)),16)))
# define _mm512_store_act(A,B)  _mm256_storeu_si256((__m256i*)(A),_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_castps_si512((B)),16)))
#endif
#else
# define _mm512_load_act(A)     _mm512_loadu_ps(A)
# define _mm512_stream_act(A,B) LIBXSMM_INTRINSICS_MM512_STREAM_PS(A,B)
# define _mm512_store_act(A,B)  _mm512_storeu_ps(A,B)
#endif

/* size variables, all const */
const int nImg = handle->desc.N;
const int ifh = handle->desc.H;
const int ifw = handle->desc.W;
const int sh = handle->desc.u;
const int sw = handle->desc.v;
const int ofh = handle->ofh;
const int ofw = handle->ofw;
const int iph = handle->desc.pad_h_in;
const int ipw = handle->desc.pad_w_in;
const int oph = handle->desc.pad_h_out;
const int opw = handle->desc.pad_w_out;
const int ofhp = ofh + 2*oph;
const int ofwp = ofw + 2*opw;
const int ifhp = ifh + 2*iph;
const int ifwp = ifw + 2*ipw;
/* here we assume that input and output blocking is similar */
const int nBlocksFm = handle->blocksifm;

/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int work = nImg * ofw * ofh * nBlocksFm;
/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* loop variables */
int img = 0;
int fm = 0;
int imgfm = 0;
int ho = 0;
int wo = 0;
int hi = 0;
int wi = 0;
int kh = 0;
int kw = 0;
int _ho = 0;
int _wo = 0;
#if defined(LIBXSMM_DNN_POOLING_FWD_AVG)
#if defined(LIBXSMM_DNN_POOLING_FWD_BF16)
float recp_pool_size = 1.0f/((float)handle->desc.R*(float)handle->desc.S);
#else
element_output_type recp_pool_size = 1.0f/((element_output_type)handle->desc.R*(element_output_type)handle->desc.S);
#endif
#endif

/* multi-dim arrays declaration */
LIBXSMM_VLA_DECL(5, const element_input_type,  input,      (element_input_type* )handle->reg_input->data,  nBlocksFm, ifhp, ifwp, 64);
LIBXSMM_VLA_DECL(5,       element_output_type, output,     (element_output_type*)handle->reg_output->data, nBlocksFm, ofhp, ofwp, 64);
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
LIBXSMM_VLA_DECL(5,       element_mask_type,   mask,       (element_mask_type*  )handle->mask->data,       nBlocksFm,  ofh,  ofw, 64);
LIBXSMM_UNUSED(mask_);
#endif

/* lazy barrier init */
for (imgfm = thr_begin; imgfm < thr_end; ++imgfm) {
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
#endif
  element_output_type*     output_ptr;
#if defined(LIBXSMM_DNN_POOLING_FWD_AVG)
  const __m512 recp_pool_size_ps = _mm512_set1_ps( recp_pool_size );
#endif
  img = imgfm / (ofw * ofh * nBlocksFm);
  fm = (imgfm % (ofw * ofh * nBlocksFm))/(ofw * ofh);
  _ho = ((imgfm % (ofw * ofh * nBlocksFm))%(ofw * ofh))/ofw;
  _wo = ((imgfm % (ofw * ofh * nBlocksFm))%(ofw * ofh))%ofw;
  ho = oph + _ho;
  wo = opw + _wo;
  wi = ((wo-opw) * sw) - handle->desc.pad_w;
  hi = ((ho-oph) * sh) - handle->desc.pad_h;

#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
#endif
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
  __m512 lcl_voutput  = _mm512_set1_ps(-FLT_MAX);
  __m512 lcl_voutput2 = _mm512_set1_ps(-FLT_MAX);
  __m512 lcl_voutput3 = _mm512_set1_ps(-FLT_MAX);
  __m512 lcl_voutput4 = _mm512_set1_ps(-FLT_MAX);
#endif
#if defined(LIBXSMM_DNN_POOLING_FWD_AVG)
  __m512 lcl_voutput  = _mm512_setzero_ps();
  __m512 lcl_voutput2 = _mm512_setzero_ps();
  __m512 lcl_voutput3 = _mm512_setzero_ps();
  __m512 lcl_voutput4 = _mm512_setzero_ps();
#endif

  for( kh = 0; kh < handle->desc.R; kh++ ) {
    if (hi+kh < 0 || hi+kh >= ifh) continue;
    for( kw = 0; kw < handle->desc.S; kw++ ) {
      if (wi+kw < 0 || wi+kw >= ifw) {
        continue;
      } else {
        const element_input_type*      input_ptr  = &LIBXSMM_VLA_ACCESS(5, input,      img, fm, hi+kh+iph, wi+kw+ipw, 0, nBlocksFm, ifhp, ifwp, 64);
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
        lcl_voutput  = _mm512_max_ps( lcl_voutput,  _mm512_load_act( input_ptr ) );
        lcl_voutput2 = _mm512_max_ps( lcl_voutput2, _mm512_load_act( input_ptr+16 ) );
        lcl_voutput3 = _mm512_max_ps( lcl_voutput3, _mm512_load_act( input_ptr+32 ) );
        lcl_voutput4 = _mm512_max_ps( lcl_voutput4, _mm512_load_act( input_ptr+48 ) );

#endif
#if defined(LIBXSMM_DNN_POOLING_FWD_AVG)
        lcl_voutput  = _mm512_add_ps( lcl_voutput,  _mm512_load_act( input_ptr ) );
        lcl_voutput2 = _mm512_add_ps( lcl_voutput2, _mm512_load_act( input_ptr+16 ) );
        lcl_voutput3 = _mm512_add_ps( lcl_voutput3, _mm512_load_act( input_ptr+32 ) );
        lcl_voutput4 = _mm512_add_ps( lcl_voutput4, _mm512_load_act( input_ptr+48 ) );
#endif
      }
    }
  }
  /* copy the local buffer into output activations */
  output_ptr = &LIBXSMM_VLA_ACCESS(5, output,     img, fm,     ho, wo, 0, nBlocksFm, ofhp, ofwp, 64);
#if defined(LIBXSMM_DNN_POOLING_FWD_AVG)
  _mm512_stream_act( output_ptr,    _mm512_mul_ps( lcl_voutput,  recp_pool_size_ps ) );
  _mm512_stream_act( output_ptr+16, _mm512_mul_ps( lcl_voutput2, recp_pool_size_ps ) );
  _mm512_stream_act( output_ptr+32, _mm512_mul_ps( lcl_voutput3, recp_pool_size_ps ) );
  _mm512_stream_act( output_ptr+48, _mm512_mul_ps( lcl_voutput4, recp_pool_size_ps ) );
#endif
#if defined(LIBXSMM_DNN_POOLING_FWD_MAX)
  _mm512_stream_act( output_ptr,    lcl_voutput );
  _mm512_stream_act( output_ptr+16, lcl_voutput2);
  _mm512_stream_act( output_ptr+32, lcl_voutput3);
  _mm512_stream_act( output_ptr+48, lcl_voutput4);
#endif
}

# undef _mm512_load_act
# undef _mm512_stream_act
# undef _mm512_store_act

