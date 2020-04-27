#ifndef TENSOR_FIXED_VIEWS_ND_H
#define TENSOR_FIXED_VIEWS_ND_H


#include "Fastor/tensor/Tensor.h"
#include "Fastor/tensor/Ranges.h"
#include "Fastor/expressions/linalg_ops/linalg_traits.h"


namespace Fastor {


// Generic const fixed tensor views based on sequences/slices
//----------------------------------------------------------------------------------------------//
template<template<typename,size_t...> class TensorType, typename T, size_t ...Rest, typename ... Fseqs>
struct TensorConstFixedViewExprnD<TensorType<T,Rest...>,Fseqs...> :
    public AbstractTensor<TensorConstFixedViewExprnD<TensorType<T,Rest...>,Fseqs...>,sizeof...(Fseqs)> {

private:
    const TensorType<T,Rest...> &expr;
    constexpr FASTOR_INLINE Tensor<T,Rest...> get_tensor() const {return expr;};
    static constexpr std::array<size_t,sizeof...(Fseqs)> products_ = nprods_views<Index<Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Fseqs)>::type>::values;
public:
    using scalar_type = T;
    using dimension_helper = get_fixed_sequence_pack_dimensions<
                            TensorType<T,Rest...>,
                            typename std_ext::make_index_sequence<sizeof...(Fseqs)>::type,
                            Fseqs...>;
    using result_type      = typename dimension_helper::type;
    using V = SIMDVector<T,DEFAULT_ABI>;
    // Get nth to_positive type
    template<int N, typename... Ts> using get_nth_pt = get_nth_type<N,to_positive_t<Ts,Rest>...>;
    static constexpr FASTOR_INDEX Dimension = sizeof...(Fseqs);
    static constexpr FASTOR_INDEX DIMS = sizeof...(Fseqs);
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {
        return dimension_helper::Size;
    }
    static constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) {
        return dimension_helper::dims[i];
    }
private:
    static constexpr std::array<int,sizeof...(Fseqs)> _dims = dimension_helper::dims;
    static constexpr bool _is_vectorisable = _dims[DIMS-1] % V::Size == 0 && (get_nth_type<DIMS-1,Fseqs...>::_step==1) ? true : false;
    static constexpr bool _is_strided_vectorisable = _dims[DIMS-1] % V::Size == 0 && (get_nth_type<DIMS-1,Fseqs...>::_step!=1) ? true : false;
public:

    constexpr FASTOR_INLINE TensorConstFixedViewExprnD(const TensorType<T,Rest...> &_ex) : expr(_ex) {}
    //----------------------------------------------------------------------------------------------//

    // Evals
    //----------------------------------------------------------------------------------------------//
    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX idx) const {

        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,DIMS> as = {};
        std::array<int,V::Size> inds;
        for (auto j=0; j<V::Size; ++j) {
            int remaining = size();
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( int(idx+j) / remaining ) % _dims[n];
            }
            inds[j] = 0;
            get_index_v<0,DIMS-1>::Do(j,inds, as);
        }
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }


    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX idx) const {

        std::array<int,DIMS> as = {};
        int remaining = size();
        for (int n = 0; n < DIMS; ++n) {
            remaining /= _dims[n];
            as[n] = ( (int)idx / remaining ) % _dims[n];
        }
        int ind = 0;
        get_index_s<0,DIMS-1>::Do(ind, as);
        return expr.eval_s(ind);
    }


    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX idx, FASTOR_INDEX j) const {

        idx += j;
        V _vec;
        std::array<int,DIMS> as = {};

        std::array<int,V::Size> inds;
        for (auto j=0; j<V::Size; ++j) {
            int remaining = size();
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( int(idx+j) / remaining ) % _dims[n];
            }
            inds[j] = 0;
            get_index_v<0,DIMS-1>::Do(j, inds, as);
        }
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }


    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX idx, FASTOR_INDEX j) const {
        idx += j;
        std::array<int,DIMS> as = {};

        int remaining = size();
        for (int n = 0; n < DIMS; ++n) {
            remaining /= _dims[n];
            as[n] = ( (int)idx / remaining ) % _dims[n];
        }
        int ind = 0;
        get_index_s<0,DIMS-1>::Do(ind, as);

        return expr.eval_s(ind);
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> teval(const std::array<int,DIMS>& as) const {
        int ind = 0;
        get_index_s<0,DIMS-1>::Do(ind, as);
        FASTOR_IF_CONSTEXPR (_is_vectorisable) return SIMDVector<T,DEFAULT_ABI>(&expr.data()[ind],false);
        else FASTOR_IF_CONSTEXPR (_is_strided_vectorisable) {
            V _vec;
            vector_setter(_vec,expr.data(),ind,get_nth_pt<DIMS-1,Fseqs...>::_step);
            return _vec;
        }
        else {

            V _vec;
            std::array<int,V::Size> inds;
            std::array<int,DIMS> as_ = as;
            for (auto j=0; j<V::Size; ++j) {
                int _sum = 0;
                get_index_s<0,DIMS-1>::Do(sum, as);
                inds[j] = _sum;

                for(int jt = (int)DIMS-1; jt>=0; jt--)
                {
                  as_[jt] +=1;
                  if(as_[jt]<_dims[jt])
                      break;
                  else
                      as_[jt]=0;
                }
            }

            vector_setter(_vec,expr.data(),inds);
            return _vec;
        }
    }

    template<typename U=T>
    FASTOR_INLINE U teval_s(const std::array<int,DIMS>& as) const {
        int ind = 0;
        get_index_s<0,DIMS-1>::Do(ind, as);
        return expr.eval_s(ind);
    }
    //----------------------------------------------------------------------------------------------//



    //----------------------------------------------------------------------------------------------//
private:
    template<size_t from, size_t to>
    struct get_index_v {
        template<size_t DIMS, size_t VSize>
        static void Do(size_t j, std::array<int,VSize> &inds, const std::array<int,DIMS>& as) {
            inds[j] += products_[from]*as[from]*get_nth_pt<from,Fseqs...>::_step + get_nth_pt<from,Fseqs...>::_first*products_[from];
            get_index_v<from+1,to>::Do(j, inds, as);
        }
    };
    template<size_t from>
    struct get_index_v<from,from> {
        template<size_t DIMS, size_t VSize>
        static void Do(size_t j, std::array<int,VSize> &inds, const std::array<int,DIMS>& as) {
            inds[j] += products_[from]*as[from]*get_nth_pt<from,Fseqs...>::_step + get_nth_pt<from,Fseqs...>::_first*products_[from];
        }
    };

    template<size_t from, size_t to>
    struct get_index_s {
        template<size_t DIMS>
        static void Do(int &ind, const std::array<int,DIMS>& as) {
            ind += products_[from]*as[from]*get_nth_pt<from,Fseqs...>::_step + get_nth_pt<from,Fseqs...>::_first*products_[from];
            get_index_s<from+1,to>::Do(ind, as);
        }
    };
    template<size_t from>
    struct get_index_s<from,from> {
        template<size_t DIMS>
        static void Do(int &ind, const std::array<int,DIMS>& as) {
            ind += products_[from]*as[from]*get_nth_pt<from,Fseqs...>::_step + get_nth_pt<from,Fseqs...>::_first*products_[from];
        }
    };
    //----------------------------------------------------------------------------------------------//
};


template<template<typename,size_t...> class TensorType, typename T, size_t ...Rest, typename ... Fseqs>
constexpr std::array<size_t,sizeof...(Fseqs)> TensorConstFixedViewExprnD<TensorType<T,Rest...>,Fseqs...>::products_;

template<template<typename,size_t...> class TensorType, typename T, size_t ...Rest, typename ... Fseqs>
constexpr std::array<int,sizeof...(Fseqs)> TensorConstFixedViewExprnD<TensorType<T,Rest...>,Fseqs...>::_dims;








//----------------------------------------------------------------------------------------------//
// Generic non-const fixed tensor views based on sequences/slices
//----------------------------------------------------------------------------------------------//
template<template<typename,size_t...> class TensorType, typename T, size_t ...Rest, typename ... Fseqs>
struct TensorFixedViewExprnD<TensorType<T,Rest...>,Fseqs...> :
    public AbstractTensor<TensorFixedViewExprnD<TensorType<T,Rest...>,Fseqs...>,sizeof...(Fseqs)> {

private:
    TensorType<T,Rest...> &expr;
    bool does_alias = false;
    constexpr FASTOR_INLINE Tensor<T,Rest...> get_tensor() const {return expr;};
    static constexpr std::array<size_t,sizeof...(Fseqs)> products_ = nprods_views<Index<Rest...>,
        typename std_ext::make_index_sequence<sizeof...(Fseqs)>::type>::values;
public:
    using scalar_type = T;
    using dimension_helper = get_fixed_sequence_pack_dimensions<
                            TensorType<T,Rest...>,
                            typename std_ext::make_index_sequence<sizeof...(Fseqs)>::type,
                            Fseqs...>;
    using result_type      = typename dimension_helper::type;
    using V = SIMDVector<T,DEFAULT_ABI>;
    // Get nth to_positive type
    template<int N, typename... Ts> using get_nth_pt = get_nth_type<N,to_positive_t<Ts,Rest>...>;
    static constexpr FASTOR_INDEX Dimension = sizeof...(Fseqs);
    static constexpr FASTOR_INDEX DIMS = sizeof...(Fseqs);
    static constexpr FASTOR_INDEX Stride = stride_finder<T>::value;
    static constexpr FASTOR_INLINE FASTOR_INDEX size() {
        return dimension_helper::Size;
    }
    static constexpr FASTOR_INLINE FASTOR_INDEX dimension(FASTOR_INDEX i) {
        return dimension_helper::dims[i];
    }
private:
    static constexpr std::array<int,sizeof...(Fseqs)> _dims = dimension_helper::dims;
    static constexpr bool _is_vectorisable = _dims[DIMS-1] % V::Size == 0 && (get_nth_type<DIMS-1,Fseqs...>::_step==1) ? true : false;
    static constexpr bool _is_strided_vectorisable = _dims[DIMS-1] % V::Size == 0 && (get_nth_type<DIMS-1,Fseqs...>::_step!=1) ? true : false;
public:

    FASTOR_INLINE TensorFixedViewExprnD<TensorType<T,Rest...>,Fseqs...>& noalias() {
        does_alias = true;
        return *this;
    }

    constexpr FASTOR_INLINE TensorFixedViewExprnD(TensorType<T,Rest...> &_ex) : expr(_ex) {}
    //----------------------------------------------------------------------------------------------//

    // View evalution operators
    // Copy assignment operators [Needed in addition to generic AbstractTensor overload]
    //----------------------------------------------------------------------------------//
    void operator=(const TensorFixedViewExprnD<TensorType<T,Rest...>,Fseqs...> &other) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorFixedViewExprnD<TensorType<T,Rest...>,Fseqs...>(tmp_this_tensor);
            // Assign other to temporary
            tmp = other;
            // assign temporary to this
            this->operator=(tmp);
            return;
        }
#endif
#ifndef NDEBUG
        FASTOR_ASSERT(other.size()==this->size(), "TENSOR SIZE MISMATCH");
        // Check if shape of tensors match
        for (FASTOR_INDEX i=0; i<Dimension; ++i) {
            FASTOR_ASSERT(other.dimension(i)==dimension(i), "TENSOR SHAPE MISMATCH");
        }
#endif

        T *_data = expr.data();
        std::array<int,DIMS> as = {};
        int total = size();
        int jt, counter = 0;

        if (_is_vectorisable) {
            // constexpr FASTOR_INDEX stride = V::Size;
            V _vec;
            while(counter < total)
            {
                int ind = 0;
                // for(int it = 0; it< DIMS; it++) {
                //     ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                // }
                get_index_s<0,DIMS-1>::Do(ind, as);
                _vec = other.template teval<T>(as);
                _vec.store(&_data[ind],false);

                counter+=V::Size;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    if (jt == _dims.size()-1) as[jt]+=V::Size;
                    else as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        else {
            while(counter < total)
            {
                int ind = 0;
                // for(int it = 0; it< DIMS; it++) {
                //     ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                // }
                get_index_s<0,DIMS-1>::Do(ind, as);
                _data[ind] = other.template teval_s<T>(as);
                // print(as);

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    as[jt] += 1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
    }
    //----------------------------------------------------------------------------------------------//

    // AbstractTensor binders [equal order]
    //----------------------------------------------------------------------------------//
    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS==DIMS && requires_evaluation_v<Derived>,bool> = false>
    void operator=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
        const typename Derived::result_type& tmp = evaluate(other.self());
        this->operator=(tmp);
    }
    template<typename Derived, size_t OTHER_DIMS, enable_if_t_<OTHER_DIMS==DIMS && !requires_evaluation_v<Derived>,bool> = false>
    void operator=(const AbstractTensor<Derived,OTHER_DIMS> &other) {
#if !(FASTOR_NO_ALIAS)
        if (does_alias) {
            does_alias = false;
            // Evaluate this into a temporary
            auto tmp_this_tensor = get_tensor();
            auto tmp = TensorFixedViewExprnD<TensorType<T,Rest...>,Fseqs...>(tmp_this_tensor);
            // Assign other to temporary
            tmp = other;
            // assign temporary to this
            this->operator=(tmp);
            return;
        }
#endif
        const Derived& other_src = other.self();
#ifndef NDEBUG
        FASTOR_ASSERT(other_src.size()==this->size(), "TENSOR SIZE MISMATCH");
#endif
        T *_data = expr.data();
        std::array<int,DIMS> as = {};
        int total = size();
        int jt, counter = 0;

        if (_is_vectorisable) {
            using V=SIMDVector<T,DEFAULT_ABI>;
            while(counter < total)
            {
                int ind = 0;
                // for(int it = 0; it< DIMS; it++) {
                //     ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                // }
                get_index_s<0,DIMS-1>::Do(ind, as);
                // V _vec = other_src.template eval<T>(counter);
                V _vec = other_src.template teval<T>(as);
                _vec.store(&_data[ind],false);

                counter+=V::Size;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    if (jt == _dims.size()-1) as[jt]+=V::Size;
                    else as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }
        }
        else
        {
            while(counter < total)
            {
                int ind = 0;
                // for(int it = 0; it< DIMS; it++) {
                //     ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
                // }
                get_index_s<0,DIMS-1>::Do(ind, as);
                // _data[ind] = other_src.template eval_s<T>(counter);
                _data[ind] = other_src.template teval_s<T>(as);

                counter++;
                for(jt = DIMS-1; jt>=0; jt--)
                {
                    as[jt] +=1;
                    if(as[jt]<_dims[jt])
                        break;
                    else
                        as[jt]=0;
                }
                if(jt<0)
                    break;
            }

            // // Generic vectorised version that takes care of the remainder scalar ops
            // using V=SIMDVector<T,DEFAULT_ABI>;
            // while(counter < total)
            // {
            //     int ind = 0;
            //     for(int it = 0; it< DIMS; it++) {
            //         ind += products_[it]*(as[it]*_seqs[it]._step + _seqs[it]._first);
            //     }
            //     get_index_s<0,DIMS-1>::Do(ind, as);
            //     if (_dims[DIMS-1] - as[DIMS-1] % V::Size == 0) {
            //         // V _vec = other_src.template eval<T>(counter);
            //         V _vec = other_src.template teval<T>(as);
            //         _vec.store(&_data[ind],false);
            //         counter+=V::Size;
            //     }
            //     else {
            //         // _data[ind] = other_src.template eval_s<T>(counter);
            //         _data[ind] = other_src.template teval_s<T>(as);
            //         counter++;
            //     }

            //     for(jt = DIMS-1; jt>=0; jt--)
            //     {
            //         if (jt == _dims.size()-1) as[jt]+=V::Size;
            //         else as[jt] +=1;
            //         if(as[jt]<_dims[jt])
            //             break;
            //         else
            //             as[jt]=0;
            //     }
            //     if(jt<0)
            //         break;
            // }
        }
    }
    //----------------------------------------------------------------------------------------------//


    // Evals
    //----------------------------------------------------------------------------------------------//
    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX idx) const {

        SIMDVector<U,DEFAULT_ABI> _vec;
        std::array<int,DIMS> as = {};
        std::array<int,V::Size> inds;
        for (auto j=0; j<V::Size; ++j) {
            int remaining = size();
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( int(idx+j) / remaining ) % _dims[n];
            }
            inds[j] = 0;
            get_index_v<0,DIMS-1>::Do(j,inds, as);
        }
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }


    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX idx) const {

        std::array<int,DIMS> as = {};
        int remaining = size();
        for (int n = 0; n < DIMS; ++n) {
            remaining /= _dims[n];
            as[n] = ( (int)idx / remaining ) % _dims[n];
        }
        int ind = 0;
        get_index_s<0,DIMS-1>::Do(ind, as);
        return expr.eval_s(ind);
    }


    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> eval(FASTOR_INDEX idx, FASTOR_INDEX j) const {

        idx += j;
        V _vec;
        std::array<int,DIMS> as = {};

        std::array<int,V::Size> inds;
        for (auto j=0; j<V::Size; ++j) {
            int remaining = size();
            for (int n = 0; n < DIMS; ++n) {
                remaining /= _dims[n];
                as[n] = ( int(idx+j) / remaining ) % _dims[n];
            }
            inds[j] = 0;
            get_index_v<0,DIMS-1>::Do(j, inds, as);
        }
        vector_setter(_vec,expr.data(),inds);
        return _vec;
    }


    template<typename U=T>
    FASTOR_INLINE U eval_s(FASTOR_INDEX idx, FASTOR_INDEX j) const {
        idx += j;
        std::array<int,DIMS> as = {};

        int remaining = size();
        for (int n = 0; n < DIMS; ++n) {
            remaining /= _dims[n];
            as[n] = ( (int)idx / remaining ) % _dims[n];
        }
        int ind = 0;
        get_index_s<0,DIMS-1>::Do(ind, as);

        return expr.eval_s(ind);
    }

    template<typename U=T>
    FASTOR_INLINE SIMDVector<U,DEFAULT_ABI> teval(const std::array<int,DIMS>& as) const {
        int ind = 0;
        get_index_s<0,DIMS-1>::Do(ind, as);
        FASTOR_IF_CONSTEXPR (_is_vectorisable) return SIMDVector<T,DEFAULT_ABI>(&expr.data()[ind],false);
        else FASTOR_IF_CONSTEXPR (_is_strided_vectorisable) {
            V _vec;
            vector_setter(_vec,expr.data(),ind,get_nth_pt<DIMS-1,Fseqs...>::_step);
            return _vec;
        }
        else {

            V _vec;
            std::array<int,V::Size> inds;
            std::array<int,DIMS> as_ = as;
            for (auto j=0; j<V::Size; ++j) {
                int _sum = 0;
                get_index_s<0,DIMS-1>::Do(sum, as);
                inds[j] = _sum;

                for(int jt = (int)DIMS-1; jt>=0; jt--)
                {
                  as_[jt] +=1;
                  if(as_[jt]<_dims[jt])
                      break;
                  else
                      as_[jt]=0;
                }
            }

            vector_setter(_vec,expr.data(),inds);
            return _vec;
        }
    }

    template<typename U=T>
    FASTOR_INLINE U teval_s(const std::array<int,DIMS>& as) const {
        int ind = 0;
        get_index_s<0,DIMS-1>::Do(ind, as);
        return expr.eval_s(ind);
    }
    //----------------------------------------------------------------------------------------------//



    //----------------------------------------------------------------------------------------------//
private:
    template<size_t from, size_t to>
    struct get_index_v {
        template<size_t DIMS, size_t VSize>
        static void Do(size_t j, std::array<int,VSize> &inds, const std::array<int,DIMS>& as) {
            inds[j] += products_[from]*as[from]*get_nth_pt<from,Fseqs...>::_step + get_nth_pt<from,Fseqs...>::_first*products_[from];
            get_index_v<from+1,to>::Do(j, inds, as);
        }
    };
    template<size_t from>
    struct get_index_v<from,from> {
        template<size_t DIMS, size_t VSize>
        static void Do(size_t j, std::array<int,VSize> &inds, const std::array<int,DIMS>& as) {
            inds[j] += products_[from]*as[from]*get_nth_pt<from,Fseqs...>::_step + get_nth_pt<from,Fseqs...>::_first*products_[from];
        }
    };

    template<size_t from, size_t to>
    struct get_index_s {
        template<size_t DIMS>
        static void Do(int &ind, const std::array<int,DIMS>& as) {
            ind += products_[from]*as[from]*get_nth_pt<from,Fseqs...>::_step + get_nth_pt<from,Fseqs...>::_first*products_[from];
            get_index_s<from+1,to>::Do(ind, as);
        }
    };
    template<size_t from>
    struct get_index_s<from,from> {
        template<size_t DIMS>
        static void Do(int &ind, const std::array<int,DIMS>& as) {
            ind += products_[from]*as[from]*get_nth_pt<from,Fseqs...>::_step + get_nth_pt<from,Fseqs...>::_first*products_[from];
        }
    };
    //----------------------------------------------------------------------------------------------//
};


template<template<typename,size_t...> class TensorType, typename T, size_t ...Rest, typename ... Fseqs>
constexpr std::array<size_t,sizeof...(Fseqs)> TensorFixedViewExprnD<TensorType<T,Rest...>,Fseqs...>::products_;

template<template<typename,size_t...> class TensorType, typename T, size_t ...Rest, typename ... Fseqs>
constexpr std::array<int,sizeof...(Fseqs)> TensorFixedViewExprnD<TensorType<T,Rest...>,Fseqs...>::_dims;







} // end of namespace Fastor


#endif // TENSOR_FIXED_VIEWS_ND_H