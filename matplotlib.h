#pragma once

#include <vector>
#include <map>
#include <array>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <functional>

#define WITH_OPENCV 1

#include <Python.h>

#ifndef WITHOUT_NUMPY
#  define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#  include <numpy/arrayobject.h>

#  ifdef WITH_OPENCV
#    include <opencv2/opencv.hpp>
#  endif // WITH_OPENCV
#endif // WITHOUT_NUMPY

#if PY_MAJOR_VERSION >= 3
#  define PyString_FromString PyUnicode_FromString
#  define PyInt_FromLong PyLong_FromLong
#  define PyString_FromString PyUnicode_FromString
#endif

namespace matplotlib {
namespace detail {

static std::string s_backend;

struct Interpreter {
    PyObject* show_function;
    PyObject* close_function;
    PyObject* draw_function;
    PyObject* pause_function;
    PyObject* save_function;
    PyObject* figure_function;
    PyObject* fignum_exists_function;
    PyObject* plot_function;
    PyObject* quiver_function;
    PyObject* semilogx_function;
    PyObject* semilogy_function;
    PyObject* loglog_function;
    PyObject* fill_function;
    PyObject* fill_between_function;
    PyObject* hist_function;
    PyObject* imshow_function;
    PyObject* scatter_function;
    PyObject* subplot_function;
    PyObject* legend_function;
    PyObject* xlim_function;
    PyObject* ion_function;
    PyObject* ginput_function;
    PyObject* ylim_function;
    PyObject* title_function;
    PyObject* axis_function;
    PyObject* xlabel_function;
    PyObject* ylabel_function;
    PyObject* xticks_function;
    PyObject* yticks_function;
    PyObject* grid_function;
    PyObject* clf_function;
    PyObject* errorbar_function;
    PyObject* annotate_function;
    PyObject* tight_layout_function;
    PyObject* colormap;
    PyObject* empty_tuple;
    PyObject* stem_function;
    PyObject* xkcd_function;
    PyObject* text_function;
    PyObject* suptitle_function;
    PyObject* bar_function;
    PyObject* subplots_adjust_function;

    static Interpreter& get() {
        static Interpreter ctx;
        return ctx;
    }

    PyObject* safe_import(PyObject* module, const std::string& name) {
        PyObject* function = PyObject_GetAttrString(module, name.c_str());

        if (!function)
            throw std::runtime_error(std::string("Couldn't find required function: ") + name);

        if (!PyFunction_Check(function))
            throw std::runtime_error(name + std::string(" is unexpectedly not a PyFunction."));

        return function;
    }

private:
#ifndef WITHOUT_NUMPY
#  if PY_MAJOR_VERSION >= 3
    void* import_numpy() {
        import_array();
        return nullptr;
    }
#  else
    void import_numpy() {
        import_array();
    }
#  endif
#endif

    Interpreter() {
        wchar_t name[] = L"plotting";
        Py_SetProgramName(name);
        Py_Initialize();

#ifndef WITHOUT_NUMPY
        import_numpy();
#endif

        PyObject* matplotlibname = PyString_FromString("matplotlib");
        PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
        PyObject* cmname = PyString_FromString("matplotlib.cm");
        PyObject* pylabname = PyString_FromString("pylab");

        if (!pyplotname || !pylabname || !matplotlibname || !cmname) {
            throw std::runtime_error("couldnt create string");
        }

        PyObject* matplotlib = PyImport_Import(matplotlibname);
        Py_DECREF(matplotlibname);
        if (!matplotlib) {
            PyErr_Print();
            throw std::runtime_error("Error loading module matplotlib!");
        }

        if (!s_backend.empty()) {
            PyObject_CallMethod(matplotlib, const_cast<char*>("use"), const_cast<char*>("s"), s_backend.c_str());
        }

        PyObject* pymod = PyImport_Import(pyplotname);
        Py_DECREF(pyplotname);
        if (!pymod) {
            throw std::runtime_error("Error loading module matplotlib.pyplot!");
        }

        colormap = PyImport_Import(cmname);
        Py_DECREF(cmname);
        if (!colormap) {
            throw std::runtime_error("Error loading module matplotlib.cm!");
        }

        PyObject* pylabmod = PyImport_Import(pylabname);
        Py_DECREF(pylabname);
        if (!pylabmod) {
            throw std::runtime_error("Error loading module pylab!");
        }

        show_function = safe_import(pymod, "show");
        close_function = safe_import(pymod, "close");
        draw_function = safe_import(pymod, "draw");
        pause_function = safe_import(pymod, "pause");
        figure_function = safe_import(pymod, "figure");
        fignum_exists_function = safe_import(pymod, "fignum_exists");
        plot_function = safe_import(pymod, "plot");
        quiver_function = safe_import(pymod, "quiver");
        semilogx_function = safe_import(pymod, "semilogx");
        semilogy_function = safe_import(pymod, "semilogy");
        loglog_function = safe_import(pymod, "loglog");
        fill_function = safe_import(pymod, "fill");
        fill_between_function = safe_import(pymod, "fill_between");
        hist_function = safe_import(pymod, "hist");
        scatter_function = safe_import(pymod, "scatter");
        subplot_function = safe_import(pymod, "subplot");
        legend_function = safe_import(pymod, "legend");
        ylim_function = safe_import(pymod, "ylim");
        title_function = safe_import(pymod, "title");
        axis_function = safe_import(pymod, "axis");
        xlabel_function = safe_import(pymod, "xlabel");
        ylabel_function = safe_import(pymod, "ylabel");
        xticks_function = safe_import(pymod, "xticks");
        yticks_function = safe_import(pymod, "yticks");
        grid_function = safe_import(pymod, "grid");
        xlim_function = safe_import(pymod, "xlim");
        ion_function = safe_import(pymod, "ion");
        ginput_function = safe_import(pymod, "ginput");
        save_function = safe_import(pylabmod, "savefig");
        annotate_function = safe_import(pymod, "annotate");
        clf_function = safe_import(pymod, "clf");
        errorbar_function = safe_import(pymod, "errorbar");
        tight_layout_function = safe_import(pymod, "tight_layout");
        stem_function = safe_import(pymod, "stem");
        xkcd_function = safe_import(pymod, "xkcd");
        text_function = safe_import(pymod, "text");
        suptitle_function = safe_import(pymod, "suptitle");
        bar_function = safe_import(pymod, "bar");
        subplots_adjust_function = safe_import(pymod, "subplots_adjust");
#ifndef WITHOUT_NUMPY
        imshow_function = safe_import(pymod, "imshow");
#endif

        empty_tuple = PyTuple_New(0);
    }

    ~Interpreter() {
        Py_Finalize();
    }
};

} // end namespace detail

// must be called before the first regular call to matplotlib to have any effect
inline void backend(const std::string& name) {
    detail::s_backend = name;
}

inline bool annotate(const std::string& annotation, double x, double y) {
    PyObject* xy = PyTuple_New(2);
    PyObject* str = PyString_FromString(annotation.c_str());

    PyTuple_SetItem(xy, 0, PyFloat_FromDouble(x));
    PyTuple_SetItem(xy, 1, PyFloat_FromDouble(y));

    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "xy", xy);

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, str);

    PyObject* res = PyObject_Call(detail::Interpreter::get().annotate_function, args, kwargs);

    Py_DECREF(args);
    Py_DECREF(kwargs);

    if (res) Py_DECREF(res);

    return res;
}

#ifndef WITHOUT_NUMPY
template <typename T>
struct SelectNpyType { const static NPY_TYPES type = NPY_NOTYPE; };

template <> struct SelectNpyType<double> { const static NPY_TYPES type = NPY_DOUBLE; };
template <> struct SelectNpyType<float> { const static NPY_TYPES type = NPY_FLOAT; };
template <> struct SelectNpyType<bool> { const static NPY_TYPES type = NPY_BOOL; };
template <> struct SelectNpyType<int8_t> { const static NPY_TYPES type = NPY_INT8; };
template <> struct SelectNpyType<int16_t> { const static NPY_TYPES type = NPY_SHORT; };
template <> struct SelectNpyType<int32_t> { const static NPY_TYPES type = NPY_INT; };
template <> struct SelectNpyType<int64_t> { const static NPY_TYPES type = NPY_INT64; };
template <> struct SelectNpyType<uint8_t> { const static NPY_TYPES type = NPY_UINT8; };
template <> struct SelectNpyType<uint16_t> { const static NPY_TYPES type = NPY_USHORT; };
template <> struct SelectNpyType<uint32_t> { const static NPY_TYPES type = NPY_ULONG; };
template <> struct SelectNpyType<uint64_t> { const static NPY_TYPES type = NPY_UINT64; };

template<typename Numeric>
PyObject* get_array(const std::vector<Numeric>& v) {
    detail::Interpreter::get();
    NPY_TYPES type = SelectNpyType<Numeric>::type;
    if (type == NPY_NOTYPE) {
        std::vector<double> vd(v.size());
        npy_intp vsize = v.size();
        std::copy(v.begin(), v.end(), vd.begin());
        PyObject* varray = PyArray_SimpleNewFromData(1, &vsize, NPY_DOUBLE, (void*)(vd.data()));
        return varray;
    }

    npy_intp vsize = v.size();
    PyObject* varray = PyArray_SimpleNewFromData(1, &vsize, type, (void*)(v.data()));
    return varray;
}

template<typename Numeric>
PyObject* get_2darray(const std::vector<::std::vector<Numeric>>& v) {
    detail::Interpreter::get();
    if (v.size() < 1) throw std::runtime_error("get_2d_array v too small");

    npy_intp vsize[2] = {static_cast<npy_intp>(v.size()),
                         static_cast<npy_intp>(v[0].size())};

    PyArrayObject* varray = (PyArrayObject*)PyArray_SimpleNew(2, vsize, NPY_DOUBLE);

    double* vd_begin = static_cast<double*>(PyArray_DATA(varray));

    for (const ::std::vector<Numeric>& v_row : v) {
        if (v_row.size() != static_cast<size_t>(vsize[1]))
            throw std::runtime_error("Mismatched array size");
        std::copy(v_row.begin(), v_row.end(), vd_begin);
        vd_begin += vsize[1];
    }

    return reinterpret_cast<PyObject*>(varray);
}

#else // fallback if we don't have numpy: copy every element of the given vector

template<typename Numeric>
PyObject* get_array(const std::vector<Numeric>& v) {
    PyObject* list = PyList_New(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        PyList_SetItem(list, i, PyFloat_FromDouble(v.at(i)));
    }
    return list;
}

#endif // WITHOUT_NUMPY

template<typename Numeric>
bool plot(const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::map<std::string, std::string>& keywords) {
    assert(x.size() == y.size());

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* args = PyTuple_New(2);
    PyTuple_SetItem(args, 0, xarray);
    PyTuple_SetItem(args, 1, yarray);

    PyObject* kwargs = PyDict_New();
    for (const auto& it : keywords) {
        PyDict_SetItemString(kwargs, it.first.c_str(), PyString_FromString(it.second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().plot_function, args, kwargs);

    Py_DECREF(args);
    Py_DECREF(kwargs);
    if (res) Py_DECREF(res);

    return res;
}

template <typename Numeric>
void plot_surface(const std::vector<::std::vector<Numeric>>& x,
                  const std::vector<::std::vector<Numeric>>& y,
                  const std::vector<::std::vector<Numeric>>& z,
                  const std::map<std::string, std::string>& keywords = {}) {
    assert(x.size() == y.size());
    assert(y.size() == z.size());

    PyObject* xarray = get_2darray(x);
    PyObject* yarray = get_2darray(y);
    PyObject* zarray = get_2darray(z);

    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, xarray);
    PyTuple_SetItem(args, 1, yarray);
    PyTuple_SetItem(args, 2, zarray);

    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "rstride", PyInt_FromLong(1));
    PyDict_SetItemString(kwargs, "cstride", PyInt_FromLong(1));

    PyObject* python_colormap_coolwarm = PyObject_GetAttrString(detail::Interpreter::get().colormap, "coolwarm");

    PyDict_SetItemString(kwargs, "cmap", python_colormap_coolwarm);

    for (const auto& it : keywords) {
        PyDict_SetItemString(kwargs, it.first.c_str(), PyString_FromString(it.second.c_str()));
    }

    PyObject* fig = PyObject_CallObject(detail::Interpreter::get().figure_function, detail::Interpreter::get().empty_tuple);
    if (!fig) throw std::runtime_error("Call to figure() failed.");

    PyObject* gca_kwargs = PyDict_New();
    PyDict_SetItemString(gca_kwargs, "projection", PyString_FromString("3d"));

    PyObject* gca = PyObject_GetAttrString(fig, "gca");
    if (!gca) throw std::runtime_error("No gca");
    Py_INCREF(gca);
    PyObject* axis = PyObject_Call(gca, detail::Interpreter::get().empty_tuple, gca_kwargs);

    if (!axis) throw std::runtime_error("No axis");
    Py_INCREF(axis);

    Py_DECREF(gca);
    Py_DECREF(gca_kwargs);

    PyObject* plot_surface = PyObject_GetAttrString(axis, "plot_surface");
    if (!plot_surface) throw std::runtime_error("No surface");
    Py_INCREF(plot_surface);
    PyObject* res = PyObject_Call(plot_surface, args, kwargs);
    if (!res) throw std::runtime_error("Failed surface");
    Py_DECREF(plot_surface);

    Py_DECREF(axis);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    if (res) Py_DECREF(res);
}

template<typename Numeric>
bool stem(const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::map<std::string, std::string>& keywords) {
    assert(x.size() == y.size());

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* args = PyTuple_New(2);
    PyTuple_SetItem(args, 0, xarray);
    PyTuple_SetItem(args, 1, yarray);

    PyObject* kwargs = PyDict_New();
    for (const auto& it : keywords) {
        PyDict_SetItemString(kwargs, it.first.c_str(), PyString_FromString(it.second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().stem_function, args, kwargs);

    Py_DECREF(args);
    Py_DECREF(kwargs);
    if (res) Py_DECREF(res);

    return res;
}

template<typename Numeric>
bool fill(const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::map<std::string, std::string>& keywords) {
    assert(x.size() == y.size());

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* args = PyTuple_New(2);
    PyTuple_SetItem(args, 0, xarray);
    PyTuple_SetItem(args, 1, yarray);

    PyObject* kwargs = PyDict_New();
    for (const auto& it : keywords) {
        PyDict_SetItemString(kwargs, it.first.c_str(), PyUnicode_FromString(it.second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().fill_function, args, kwargs);

    Py_DECREF(args);
    Py_DECREF(kwargs);

    if (res) Py_DECREF(res);

    return res;
}

template<typename Numeric>
bool fill_between(const std::vector<Numeric>& x, const std::vector<Numeric>& y1, const std::vector<Numeric>& y2, const std::map<std::string, std::string>& keywords) {
    assert(x.size() == y1.size());
    assert(x.size() == y2.size());

    PyObject* xarray = get_array(x);
    PyObject* y1array = get_array(y1);
    PyObject* y2array = get_array(y2);

    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, xarray);
    PyTuple_SetItem(args, 1, y1array);
    PyTuple_SetItem(args, 2, y2array);

    PyObject* kwargs = PyDict_New();
    for (const auto& it : keywords) {
        PyDict_SetItemString(kwargs, it.first.c_str(), PyUnicode_FromString(it.second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().fill_between_function, args, kwargs);

    Py_DECREF(args);
    Py_DECREF(kwargs);
    if (res) Py_DECREF(res);

    return res;
}

template<typename Numeric>
bool hist(const std::vector<Numeric>& y, long bins = 10, std::string color = "b", double alpha = 1.0, bool cumulative = false) {
    PyObject* yarray = get_array(y);

    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "bins", PyLong_FromLong(bins));
    PyDict_SetItemString(kwargs, "color", PyString_FromString(color.c_str()));
    PyDict_SetItemString(kwargs, "alpha", PyFloat_FromDouble(alpha));
    PyDict_SetItemString(kwargs, "cumulative", cumulative ? Py_True : Py_False);

    PyObject* plot_args = PyTuple_New(1);
    PyTuple_SetItem(plot_args, 0, yarray);

    PyObject* res = PyObject_Call(detail::Interpreter::get().hist_function, plot_args, kwargs);

    Py_DECREF(plot_args);
    Py_DECREF(kwargs);
    if (res) Py_DECREF(res);

    return res;
}

#ifndef WITHOUT_NUMPY
namespace internal {
void imshow(void* ptr, const NPY_TYPES type, const int rows, const int columns, const int colors, const std::map<std::string, std::string>& keywords) {
    assert(type == NPY_UINT8 || type == NPY_FLOAT);
    assert(colors == 1 || colors == 3 || colors == 4);

    detail::Interpreter::get();

    npy_intp dims[3] = { rows, columns, colors };
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyArray_SimpleNewFromData(colors == 1 ? 2 : 3, dims, type, ptr));

    PyObject* kwargs = PyDict_New();
    for (const auto& it : keywords) {
        PyDict_SetItemString(kwargs, it.first.c_str(), PyUnicode_FromString(it.second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().imshow_function, args, kwargs);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    if (!res) throw std::runtime_error("Call to imshow() failed");
    Py_DECREF(res);
}
} // namespace internal

inline void imshow(const unsigned char* ptr, const int rows, const int columns, const int colors, const std::map<std::string, std::string>& keywords = {}) {
    internal::imshow((void*)ptr, NPY_UINT8, rows, columns, colors, keywords);
}

inline void imshow(const float* ptr, const int rows, const int columns, const int colors, const std::map<std::string, std::string>& keywords = {}) {
    internal::imshow((void*)ptr, NPY_FLOAT, rows, columns, colors, keywords);
}

#ifdef WITH_OPENCV
void imshow(const cv::Mat& image, const std::map<std::string, std::string>& keywords = {}) {
    cv::Mat image2;
    NPY_TYPES npy_type = NPY_UINT8;

    switch (image.type() & CV_MAT_DEPTH_MASK) {
    case CV_8U:
        image2 = image.clone();
        break;
    case CV_32F:
        image2 = image.clone();
        npy_type = NPY_FLOAT;
        break;
    default:
        image.convertTo(image2, CV_MAKETYPE(CV_8U, image.channels()));
    }

    switch (image2.channels()) {
    case 3:
        cv::cvtColor(image2, image2, cv::COLOR_BGR2RGB);
        break;
    case 4:
        cv::cvtColor(image2, image2, cv::COLOR_BGRA2RGBA);
    }

    internal::imshow(image2.data, npy_type, image2.rows, image2.cols, image2.channels(), keywords);
}
#endif // WITH_OPENCV
#endif // WITHOUT_NUMPY

template<typename NumericX, typename NumericY>
bool scatter(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const double s = 1.0) {
    assert(x.size() == y.size());

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "s", PyLong_FromLong(s));

    PyObject* plot_args = PyTuple_New(2);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);

    PyObject* res = PyObject_Call(detail::Interpreter::get().scatter_function, plot_args, kwargs);

    Py_DECREF(plot_args);
    Py_DECREF(kwargs);
    if (res) Py_DECREF(res);

    return res;
}

template<typename Numeric>
bool bar(const std::vector<Numeric>& y, std::string ec = "black", std::string ls = "-", double lw = 1.0, const std::map<std::string, std::string>& keywords = {}) {
    PyObject* yarray = get_array(y);

    std::vector<int> x;
    for (int i = 0; i < y.size(); i++)
        x.push_back(i);

    PyObject* xarray = get_array(x);

    PyObject* kwargs = PyDict_New();

    PyDict_SetItemString(kwargs, "ec", PyString_FromString(ec.c_str()));
    PyDict_SetItemString(kwargs, "ls", PyString_FromString(ls.c_str()));
    PyDict_SetItemString(kwargs, "lw", PyFloat_FromDouble(lw));

    PyObject* plot_args = PyTuple_New(2);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);

    PyObject* res = PyObject_Call(detail::Interpreter::get().bar_function, plot_args, kwargs);

    Py_DECREF(plot_args);
    Py_DECREF(kwargs);
    if (res) Py_DECREF(res);

    return res;
}

inline bool subplots_adjust(const std::map<std::string, double>& keywords = {}) {
    PyObject* kwargs = PyDict_New();
    for (const auto& it : keywords) {
        PyDict_SetItemString(kwargs, it.first.c_str(), PyFloat_FromDouble(it.second));
    }

    PyObject* plot_args = PyTuple_New(0);

    PyObject* res = PyObject_Call(detail::Interpreter::get().subplots_adjust_function, plot_args, kwargs);

    Py_DECREF(plot_args);
    Py_DECREF(kwargs);
    if (res) Py_DECREF(res);

    return res;
}

template<typename Numeric>
bool named_hist(std::string label, const std::vector<Numeric>& y, long bins = 10, std::string color = "b", double alpha = 1.0) {
    PyObject* yarray = get_array(y);

    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
    PyDict_SetItemString(kwargs, "bins", PyLong_FromLong(bins));
    PyDict_SetItemString(kwargs, "color", PyString_FromString(color.c_str()));
    PyDict_SetItemString(kwargs, "alpha", PyFloat_FromDouble(alpha));

    PyObject* plot_args = PyTuple_New(1);
    PyTuple_SetItem(plot_args, 0, yarray);

    PyObject* res = PyObject_Call(detail::Interpreter::get().hist_function, plot_args, kwargs);

    Py_DECREF(plot_args);
    Py_DECREF(kwargs);
    if (res) Py_DECREF(res);

    return res;
}

template<typename NumericX, typename NumericY>
bool plot(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "") {
    assert(x.size() == y.size());

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(s.c_str());

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, pystring);

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().plot_function, plot_args);

    Py_DECREF(plot_args);
    if (res) Py_DECREF(res);

    return res;
}

template<typename NumericX, typename NumericY, typename NumericU, typename NumericW>
bool quiver(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::vector<NumericU>& u, const std::vector<NumericW>& w, const std::map<std::string, std::string>& keywords = {}) {
    assert(x.size() == y.size() && x.size() == u.size() && u.size() == w.size());

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);
    PyObject* uarray = get_array(u);
    PyObject* warray = get_array(w);

    PyObject* plot_args = PyTuple_New(4);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, uarray);
    PyTuple_SetItem(plot_args, 3, warray);

    PyObject* kwargs = PyDict_New();
    for (const auto& it : keywords) {
        PyDict_SetItemString(kwargs, it.first.c_str(), PyUnicode_FromString(it.second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().quiver_function, plot_args, kwargs);

    Py_DECREF(kwargs);
    Py_DECREF(plot_args);
    if (res) Py_DECREF(res);

    return res;
}

template<typename NumericX, typename NumericY>
bool stem(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "") {
    assert(x.size() == y.size());

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(s.c_str());

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, pystring);

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().stem_function, plot_args);

    Py_DECREF(plot_args);
    if (res) Py_DECREF(res);

    return res;
}

template<typename NumericX, typename NumericY>
bool semilogx(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "") {
    assert(x.size() == y.size());

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(s.c_str());

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, pystring);

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().semilogx_function, plot_args);

    Py_DECREF(plot_args);
    if (res) Py_DECREF(res);

    return res;
}

template<typename NumericX, typename NumericY>
bool semilogy(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "") {
    assert(x.size() == y.size());

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(s.c_str());

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, pystring);

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().semilogy_function, plot_args);

    Py_DECREF(plot_args);
    if (res) Py_DECREF(res);

    return res;
}

template<typename NumericX, typename NumericY>
bool loglog(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "") {
    assert(x.size() == y.size());

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(s.c_str());

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, pystring);

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().loglog_function, plot_args);

    Py_DECREF(plot_args);
    if (res) Py_DECREF(res);

    return res;
}

template<typename NumericX, typename NumericY>
bool errorbar(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::vector<NumericX>& yerr, const std::map<std::string, std::string>& keywords = {}) {
    assert(x.size() == y.size());

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);
    PyObject* yerrarray = get_array(yerr);

    PyObject* kwargs = PyDict_New();
    for (const auto& it : keywords) {
        PyDict_SetItemString(kwargs, it.first.c_str(), PyString_FromString(it.second.c_str()));
    }

    PyDict_SetItemString(kwargs, "yerr", yerrarray);

    PyObject* plot_args = PyTuple_New(2);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);

    PyObject* res = PyObject_Call(detail::Interpreter::get().errorbar_function, plot_args, kwargs);

    Py_DECREF(kwargs);
    Py_DECREF(plot_args);

    if (res)
        Py_DECREF(res);
    else
        throw std::runtime_error("Call to errorbar() failed.");

    return res;
}

template<typename Numeric>
bool named_plot(const std::string& name, const std::vector<Numeric>& y, const std::string& format = "") {
    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(format.c_str());

    PyObject* plot_args = PyTuple_New(2);
    PyTuple_SetItem(plot_args, 0, yarray);
    PyTuple_SetItem(plot_args, 1, pystring);

    PyObject* res = PyObject_Call(detail::Interpreter::get().plot_function, plot_args, kwargs);

    Py_DECREF(kwargs);
    Py_DECREF(plot_args);
    if (res) Py_DECREF(res);

    return res;
}

template<typename Numeric>
bool named_plot(const std::string& name, const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::string& format = "") {
    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(format.c_str());

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, pystring);

    PyObject* res = PyObject_Call(detail::Interpreter::get().plot_function, plot_args, kwargs);

    Py_DECREF(kwargs);
    Py_DECREF(plot_args);
    if (res) Py_DECREF(res);

    return res;
}

template<typename Numeric>
bool named_semilogx(const std::string& name, const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::string& format = "") {
    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(format.c_str());

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, pystring);

    PyObject* res = PyObject_Call(detail::Interpreter::get().semilogx_function, plot_args, kwargs);

    Py_DECREF(kwargs);
    Py_DECREF(plot_args);
    if (res) Py_DECREF(res);

    return res;
}

template<typename Numeric>
bool named_semilogy(const std::string& name, const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::string& format = "") {
    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(format.c_str());

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, pystring);

    PyObject* res = PyObject_Call(detail::Interpreter::get().semilogy_function, plot_args, kwargs);

    Py_DECREF(kwargs);
    Py_DECREF(plot_args);
    if (res) Py_DECREF(res);

    return res;
}

template<typename Numeric>
bool named_loglog(const std::string& name, const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::string& format = "") {
    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(format.c_str());

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, pystring);

    PyObject* res = PyObject_Call(detail::Interpreter::get().loglog_function, plot_args, kwargs);

    Py_DECREF(kwargs);
    Py_DECREF(plot_args);
    if (res) Py_DECREF(res);

    return res;
}

template<typename Numeric>
bool plot(const std::vector<Numeric>& y, const std::string& format = "") {
    std::vector<Numeric> x(y.size());
    for (size_t i = 0; i < x.size(); ++i) x[i] = i;
    return plot(x, y, format);
}

template<typename Numeric>
bool plot(const std::vector<Numeric>& y, const std::map<std::string, std::string>& keywords) {
    std::vector<Numeric> x(y.size());
    for (size_t i = 0; i < x.size(); ++i) x[i] = i;
    return plot(x, y, keywords);
}

template<typename Numeric>
bool stem(const std::vector<Numeric>& y, const std::string& format = "") {
    std::vector<Numeric> x(y.size());
    for (size_t i = 0; i < x.size(); ++i) x[i] = i;
    return stem(x, y, format);
}

template<typename Numeric>
void text(Numeric x, Numeric y, const std::string& s = "") {
    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, PyFloat_FromDouble(x));
    PyTuple_SetItem(args, 1, PyFloat_FromDouble(y));
    PyTuple_SetItem(args, 2, PyString_FromString(s.c_str()));

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().text_function, args);
    if (!res) throw std::runtime_error("Call to text() failed.");

    Py_DECREF(args);
    Py_DECREF(res);
}

inline long figure(long number = -1) {
    PyObject* res;
    if (number == -1)
        res = PyObject_CallObject(detail::Interpreter::get().figure_function, detail::Interpreter::get().empty_tuple);
    else {
        assert(number > 0);

        detail::Interpreter::get();

        PyObject* args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, PyLong_FromLong(number));
        res = PyObject_CallObject(detail::Interpreter::get().figure_function, args);
        Py_DECREF(args);
    }

    if (!res) throw std::runtime_error("Call to figure() failed.");

    PyObject* num = PyObject_GetAttrString(res, "number");
    if (!num) throw std::runtime_error("Could not get number attribute of figure object");
    const long figureNumber = PyLong_AsLong(num);

    Py_DECREF(num);
    Py_DECREF(res);

    return figureNumber;
}

inline bool fignum_exists(long number) {
    detail::Interpreter::get();

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyLong_FromLong(number));
    PyObject* res = PyObject_CallObject(detail::Interpreter::get().fignum_exists_function, args);
    if (!res) throw std::runtime_error("Call to fignum_exists() failed.");

    bool ret = PyObject_IsTrue(res);
    Py_DECREF(res);
    Py_DECREF(args);

    return ret;
}

inline void figure_size(size_t w, size_t h) {
    detail::Interpreter::get();

    const size_t dpi = 100;
    PyObject* size = PyTuple_New(2);
    PyTuple_SetItem(size, 0, PyFloat_FromDouble((double)w / dpi));
    PyTuple_SetItem(size, 1, PyFloat_FromDouble((double)h / dpi));

    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "figsize", size);
    PyDict_SetItemString(kwargs, "dpi", PyLong_FromSize_t(dpi));

    PyObject* res = PyObject_Call(detail::Interpreter::get().figure_function, detail::Interpreter::get().empty_tuple, kwargs);Py_DECREF(kwargs);

    if (!res) throw std::runtime_error("Call to figure_size() failed.");
    Py_DECREF(res);
}

inline void legend() {
    PyObject* res = PyObject_CallObject(detail::Interpreter::get().legend_function, detail::Interpreter::get().empty_tuple);
    if (!res) throw std::runtime_error("Call to legend() failed.");

    Py_DECREF(res);
}

template<typename Numeric>
void ylim(Numeric left, Numeric right) {
    PyObject* list = PyList_New(2);
    PyList_SetItem(list, 0, PyFloat_FromDouble(left));
    PyList_SetItem(list, 1, PyFloat_FromDouble(right));

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, list);

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().ylim_function, args);
    if (!res) throw std::runtime_error("Call to ylim() failed.");

    Py_DECREF(args);
    Py_DECREF(res);
}

template<typename Numeric>
void xlim(Numeric left, Numeric right) {
    PyObject* list = PyList_New(2);
    PyList_SetItem(list, 0, PyFloat_FromDouble(left));
    PyList_SetItem(list, 1, PyFloat_FromDouble(right));

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, list);

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().xlim_function, args);
    if (!res) throw std::runtime_error("Call to xlim() failed.");

    Py_DECREF(args);
    Py_DECREF(res);
}

inline double* xlim() {
    PyObject* args = PyTuple_New(0);
    PyObject* res = PyObject_CallObject(detail::Interpreter::get().xlim_function, args);
    PyObject* left = PyTuple_GetItem(res, 0);
    PyObject* right = PyTuple_GetItem(res, 1);

    double* arr = new double[2];
    arr[0] = PyFloat_AsDouble(left);
    arr[1] = PyFloat_AsDouble(right);

    if (!res) throw std::runtime_error("Call to xlim() failed.");

    Py_DECREF(res);
    return arr;
}

inline double* ylim() {
    PyObject* args = PyTuple_New(0);
    PyObject* res = PyObject_CallObject(detail::Interpreter::get().ylim_function, args);
    PyObject* left = PyTuple_GetItem(res, 0);
    PyObject* right = PyTuple_GetItem(res, 1);

    double* arr = new double[2];
    arr[0] = PyFloat_AsDouble(left);
    arr[1] = PyFloat_AsDouble(right);

    if (!res) throw std::runtime_error("Call to ylim() failed.");

    Py_DECREF(res);
    return arr;
}

template<typename Numeric>
inline void xticks(const std::vector<Numeric>& ticks, const std::vector<std::string>& labels = {}, const std::map<std::string, std::string>& keywords = {}) {
    assert(labels.size() == 0 || ticks.size() == labels.size());

    PyObject* ticksarray = get_array(ticks);

    PyObject* args;
    if (labels.size() == 0) {
        args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, ticksarray);
    } else {
        PyObject* labelstuple = PyTuple_New(labels.size());
        for (size_t i = 0; i < labels.size(); i++)
            PyTuple_SetItem(labelstuple, i, PyUnicode_FromString(labels[i].c_str()));

        args = PyTuple_New(2);
        PyTuple_SetItem(args, 0, ticksarray);
        PyTuple_SetItem(args, 1, labelstuple);
    }

    PyObject* kwargs = PyDict_New();
    for (const auto& it : keywords) {
        PyDict_SetItemString(kwargs, it.first.c_str(), PyString_FromString(it.second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().xticks_function, args, kwargs);

    Py_DECREF(args);
    Py_DECREF(kwargs);
    if (!res) throw std::runtime_error("Call to xticks() failed");

    Py_DECREF(res);
}

template<typename Numeric>
inline void xticks(const std::vector<Numeric>& ticks, const std::map<std::string, std::string>& keywords) {
    xticks(ticks, {}, keywords);
}

template<typename Numeric>
inline void yticks(const std::vector<Numeric>& ticks, const std::vector<std::string>& labels = {}, const std::map<std::string, std::string>& keywords = {}) {
    assert(labels.size() == 0 || ticks.size() == labels.size());

    PyObject* ticksarray = get_array(ticks);

    PyObject* args;
    if (labels.size() == 0) {
        args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, ticksarray);
    } else {
        PyObject* labelstuple = PyTuple_New(labels.size());
        for (size_t i = 0; i < labels.size(); i++)
            PyTuple_SetItem(labelstuple, i, PyUnicode_FromString(labels[i].c_str()));

        args = PyTuple_New(2);
        PyTuple_SetItem(args, 0, ticksarray);
        PyTuple_SetItem(args, 1, labelstuple);
    }

    PyObject* kwargs = PyDict_New();
    for (const auto& it : keywords) {
        PyDict_SetItemString(kwargs, it.first.c_str(), PyString_FromString(it.second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().yticks_function, args, kwargs);

    Py_DECREF(args);
    Py_DECREF(kwargs);
    if (!res) throw std::runtime_error("Call to yticks() failed");

    Py_DECREF(res);
}

template<typename Numeric>
inline void yticks(const std::vector<Numeric>& ticks, const std::map<std::string, std::string>& keywords) {
    yticks(ticks, {}, keywords);
}

inline void subplot(long nrows, long ncols, long plot_number) {
    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, PyFloat_FromDouble(nrows));
    PyTuple_SetItem(args, 1, PyFloat_FromDouble(ncols));
    PyTuple_SetItem(args, 2, PyFloat_FromDouble(plot_number));

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().subplot_function, args);
    if (!res) throw std::runtime_error("Call to subplot() failed.");

    Py_DECREF(args);
    Py_DECREF(res);
}

inline void title(const std::string& titlestr, const std::map<std::string, std::string>& keywords = {}) {
    PyObject* pytitlestr = PyString_FromString(titlestr.c_str());
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, pytitlestr);

    PyObject* kwargs = PyDict_New();
    for (const auto& it : keywords) {
        PyDict_SetItemString(kwargs, it.first.c_str(), PyUnicode_FromString(it.second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().title_function, args, kwargs);
    if (!res) throw std::runtime_error("Call to title() failed.");

    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(res);
}

inline void suptitle(const std::string& suptitlestr, const std::map<std::string, std::string>& keywords = {}) {
    PyObject* pysuptitlestr = PyString_FromString(suptitlestr.c_str());
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, pysuptitlestr);

    PyObject* kwargs = PyDict_New();
    for (const auto& it : keywords) {
        PyDict_SetItemString(kwargs, it.first.c_str(), PyUnicode_FromString(it.second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().suptitle_function, args, kwargs);
    if (!res) throw std::runtime_error("Call to suptitle() failed.");

    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(res);
}

inline void axis(const std::string& axisstr) {
    PyObject* str = PyString_FromString(axisstr.c_str());
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, str);

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().axis_function, args);
    if (!res) throw std::runtime_error("Call to axis() failed.");

    Py_DECREF(args);
    Py_DECREF(res);
}

inline void xlabel(const std::string& str, const std::map<std::string, std::string>& keywords = {}) {
    PyObject* pystr = PyString_FromString(str.c_str());
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, pystr);

    PyObject* kwargs = PyDict_New();
    for (const auto& it : keywords) {
        PyDict_SetItemString(kwargs, it.first.c_str(), PyUnicode_FromString(it.second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().xlabel_function, args, kwargs);
    if (!res) throw std::runtime_error("Call to xlabel() failed.");

    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(res);
}

inline void ylabel(const std::string& str, const std::map<std::string, std::string>& keywords = {}) {
    PyObject* pystr = PyString_FromString(str.c_str());
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, pystr);

    PyObject* kwargs = PyDict_New();
    for (const auto& it : keywords) {
        PyDict_SetItemString(kwargs, it.first.c_str(), PyUnicode_FromString(it.second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().ylabel_function, args, kwargs);
    if (!res) throw std::runtime_error("Call to ylabel() failed.");

    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(res);
}

inline void grid(bool flag) {
    PyObject* pyflag = flag ? Py_True : Py_False;
    Py_INCREF(pyflag);

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, pyflag);

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().grid_function, args);
    if (!res) throw std::runtime_error("Call to grid() failed.");

    Py_DECREF(args);
    Py_DECREF(res);
}

inline void show(const bool block = true) {
    PyObject* res;
    if (block) {
        res = PyObject_CallObject(detail::Interpreter::get().show_function, detail::Interpreter::get().empty_tuple);
    } else {
        PyObject* kwargs = PyDict_New();
        PyDict_SetItemString(kwargs, "block", Py_False);
        res = PyObject_Call(detail::Interpreter::get().show_function, detail::Interpreter::get().empty_tuple, kwargs);
        Py_DECREF(kwargs);
    }

    if (!res) throw std::runtime_error("Call to show() failed.");

    Py_DECREF(res);
}

inline void close() {
    PyObject* res = PyObject_CallObject(detail::Interpreter::get().close_function, detail::Interpreter::get().empty_tuple);

    if (!res) throw std::runtime_error("Call to close() failed.");

    Py_DECREF(res);
}

inline void xkcd() {
    PyObject* res;
    PyObject* kwargs = PyDict_New();

    res = PyObject_Call(detail::Interpreter::get().xkcd_function, detail::Interpreter::get().empty_tuple, kwargs);

    Py_DECREF(kwargs);

    if (!res) throw std::runtime_error("Call to xkcd() failed.");

    Py_DECREF(res);
}

inline void draw() {
    PyObject* res = PyObject_CallObject(detail::Interpreter::get().draw_function, detail::Interpreter::get().empty_tuple);

    if (!res) throw std::runtime_error("Call to draw() failed.");

    Py_DECREF(res);
}

template<typename Numeric>
inline void pause(Numeric interval) {
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyFloat_FromDouble(interval));

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().pause_function, args);
    if (!res) throw std::runtime_error("Call to pause() failed.");

    Py_DECREF(args);
    Py_DECREF(res);
}

inline void save(const std::string& filename) {
    PyObject* pyfilename = PyString_FromString(filename.c_str());

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, pyfilename);

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().save_function, args);
    if (!res) throw std::runtime_error("Call to save() failed.");

    Py_DECREF(args);
    Py_DECREF(res);
}

inline void clf() {
    PyObject* res = PyObject_CallObject(detail::Interpreter::get().clf_function, detail::Interpreter::get().empty_tuple);

    if (!res) throw std::runtime_error("Call to clf() failed.");

    Py_DECREF(res);
}

inline void ion() {
    PyObject* res = PyObject_CallObject(detail::Interpreter::get().ion_function, detail::Interpreter::get().empty_tuple);

    if (!res) throw std::runtime_error("Call to ion() failed.");

    Py_DECREF(res);
}

inline std::vector<std::array<double, 2>> ginput(const int numClicks = 1, const std::map<std::string, std::string>& keywords = {}) {
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyLong_FromLong(numClicks));

    PyObject* kwargs = PyDict_New();
    for (const auto& it : keywords) {
        PyDict_SetItemString(kwargs, it.first.c_str(), PyUnicode_FromString(it.second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().ginput_function, args, kwargs);

    Py_DECREF(kwargs);
    Py_DECREF(args);
    if (!res) throw std::runtime_error("Call to ginput() failed.");

    const size_t len = PyList_Size(res);
    std::vector<std::array<double, 2>> out;
    out.reserve(len);
    for (size_t i = 0; i < len; i++) {
        PyObject* current = PyList_GetItem(res, i);
        std::array<double, 2> position;
        position[0] = PyFloat_AsDouble(PyTuple_GetItem(current, 0));
        position[1] = PyFloat_AsDouble(PyTuple_GetItem(current, 1));
        out.push_back(position);
    }
    Py_DECREF(res);

    return out;
}

inline void tight_layout() {
    PyObject* res = PyObject_CallObject(detail::Interpreter::get().tight_layout_function, detail::Interpreter::get().empty_tuple);

    if (!res) throw std::runtime_error("Call to tight_layout() failed.");

    Py_DECREF(res);
}

namespace detail {

template<typename T>
using is_function = typename std::is_function<std::remove_pointer<std::remove_reference<T>>>::type;

template<bool obj, typename T>
struct is_callable_impl;

template<typename T>
struct is_callable_impl<false, T> {
    typedef is_function<T> type;
};

template<typename T>
struct is_callable_impl<true, T> {
    struct Fallback { void operator()(); };
    struct Derived : T, Fallback { };

    template<typename U, U> struct Check;

    template<typename U>
    static std::true_type test(...);

    template<typename U>
    static std::false_type test(Check<void(Fallback::*)(), &U::operator()>*);

public:
    typedef decltype(test<Derived>(nullptr)) type;
    typedef decltype(&Fallback::operator()) dtype;
    static constexpr bool value = type::value;
};

template<typename T>
struct is_callable {
    typedef typename is_callable_impl<std::is_class<T>::value, T>::type type;
};

template<typename IsYDataCallable>
struct plot_impl { };

template<>
struct plot_impl<std::false_type> {
    template<typename IterableX, typename IterableY>
    bool operator()(const IterableX& x, const IterableY& y, const std::string& format) {
        using std::begin;
        using std::end;

        auto xs = distance(begin(x), end(x));
        auto ys = distance(begin(y), end(y));
        assert(xs == ys && "x and y data must have the same number of elements!");

        PyObject* xlist = PyList_New(xs);
        PyObject* ylist = PyList_New(ys);
        PyObject* pystring = PyString_FromString(format.c_str());

        auto itx = begin(x);
        auto ity = begin(y);
        for (size_t i = 0; i < xs; ++i) {
            PyList_SetItem(xlist, i, PyFloat_FromDouble(*itx++));
            PyList_SetItem(ylist, i, PyFloat_FromDouble(*ity++));
        }

        PyObject* plot_args = PyTuple_New(3);
        PyTuple_SetItem(plot_args, 0, xlist);
        PyTuple_SetItem(plot_args, 1, ylist);
        PyTuple_SetItem(plot_args, 2, pystring);

        PyObject* res = PyObject_CallObject(detail::Interpreter::get().plot_function, plot_args);

        Py_DECREF(plot_args);
        if (res) Py_DECREF(res);

        return res;
    }
};

template<>
struct plot_impl<std::true_type> {
    template<typename Iterable, typename Callable>
    bool operator()(const Iterable& ticks, const Callable& f, const std::string& format) {
        if (begin(ticks) == end(ticks)) return true;

        std::vector<double> y;
        for (auto x : ticks) y.push_back(f(x));
        return plot_impl<std::false_type>()(ticks, y, format);
    }
};

} // end namespace detail

template<typename... Args>
bool plot() { return true; }

template<typename A, typename B, typename... Args>
bool plot(const A& a, const B& b, const std::string& format, Args... args) {
    return detail::plot_impl<typename detail::is_callable<B>::type>()(a, b, format) && plot(args...);
}

inline bool plot(const std::vector<double>& x, const std::vector<double>& y, const std::string& format = "") {
    return plot<double, double>(x, y, format);
}

inline bool plot(const std::vector<double>& y, const std::string& format = "") {
    return plot<double>(y, format);
}

inline bool plot(const std::vector<double>& x, const std::vector<double>& y, const std::map<std::string, std::string>& keywords) {
    return plot<double>(x, y, keywords);
}

class Plot {
public:
    template<typename Numeric>
    Plot(const std::string& name, const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::string& format = "") {
        assert(x.size() == y.size());

        PyObject* kwargs = PyDict_New();
        if (!name.empty()) PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

        PyObject* xarray = get_array(x);
        PyObject* yarray = get_array(y);

        PyObject* pystring = PyString_FromString(format.c_str());

        PyObject* plot_args = PyTuple_New(3);
        PyTuple_SetItem(plot_args, 0, xarray);
        PyTuple_SetItem(plot_args, 1, yarray);
        PyTuple_SetItem(plot_args, 2, pystring);

        PyObject* res = PyObject_Call(detail::Interpreter::get().plot_function, plot_args, kwargs);

        Py_DECREF(kwargs);
        Py_DECREF(plot_args);

        if (res) {
            line = PyList_GetItem(res, 0);

            if (line) {
                set_data_fct = PyObject_GetAttrString(line, "set_data");
            } else {
                Py_DECREF(line);
            }
            Py_DECREF(res);
        }
    }

    Plot(const std::string& name = "", const std::string& format = "")
        : Plot(name, std::vector<double>(), std::vector<double>(), format) {}

    template<typename Numeric>
    bool update(const std::vector<Numeric>& x, const std::vector<Numeric>& y) {
        assert(x.size() == y.size());
        if (set_data_fct) {
            PyObject* xarray = get_array(x);
            PyObject* yarray = get_array(y);

            PyObject* plot_args = PyTuple_New(2);
            PyTuple_SetItem(plot_args, 0, xarray);
            PyTuple_SetItem(plot_args, 1, yarray);

            PyObject* res = PyObject_CallObject(set_data_fct, plot_args);
            if (res) Py_DECREF(res);
            return res;
        }
        return false;
    }

    bool clear() {
        return update(std::vector<double>(), std::vector<double>());
    }

    void remove() {
        if (line) {
            auto remove_fct = PyObject_GetAttrString(line, "remove");
            PyObject* args = PyTuple_New(0);
            PyObject* res = PyObject_CallObject(remove_fct, args);
            if (res) Py_DECREF(res);
        }
        decref();
    }

    ~Plot() {
        decref();
    }
private:
    void decref() {
        if (line) Py_DECREF(line);
        if (set_data_fct) Py_DECREF(set_data_fct);
    }

    PyObject* line = nullptr;
    PyObject* set_data_fct = nullptr;
};

} // end namespace matplotlib