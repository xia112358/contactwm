#include <boost/python.hpp>
#include <coal/distance.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace bp = boost::python;
struct Buffer {
  Py_buffer view{};
  explicit Buffer(bp::object obj, bool writable=false) {
    if (PyObject_GetBuffer(obj.ptr(), &view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT |
                          (writable ? PyBUF_WRITABLE : 0)) != 0) bp::throw_error_already_set();
  }
  ~Buffer() { PyBuffer_Release(&view); }
  Buffer(const Buffer&) = delete;
};

bp::object run(bp::list geometries, bp::object positions, bp::object rotations,
               bp::object candidates, const coal::DistanceRequest& request) {
  Buffer p(positions), r(rotations), c(candidates);
  const auto is_double = [](const Py_buffer& v) {
    return v.itemsize == sizeof(double) && std::strcmp(v.format, "d") == 0;
  };
  const auto is_int64 = [](const Py_buffer& v) {
    return v.itemsize == sizeof(std::int64_t) &&
      (std::strcmp(v.format, "l") == 0 || std::strcmp(v.format, "q") == 0);
  };
  if (!is_double(p.view) || !is_double(r.view) || !is_int64(c.view))
    throw std::invalid_argument("Expected native float64 poses and int64 candidates");
  if (p.view.ndim != 3 || p.view.shape[2] != 3 || p.view.itemsize != 8 ||
      r.view.ndim != 4 || r.view.shape[0] != p.view.shape[0] ||
      r.view.shape[1] != p.view.shape[1] || r.view.shape[2] != 3 || r.view.shape[3] != 3 ||
      r.view.itemsize != 8 || c.view.ndim != 2 || c.view.shape[1] != 3 || c.view.itemsize != 8)
    throw std::invalid_argument("Invalid batch geometry buffer shape/type");
  const auto batch=p.view.shape[0], shapes=p.view.shape[1], count=c.view.shape[0];
  if (bp::len(geometries) != shapes) throw std::invalid_argument("Geometry count mismatch");
  std::vector<const coal::CollisionGeometry*> geometry;
  for (Py_ssize_t i=0; i<shapes; ++i)
    geometry.push_back(bp::extract<const coal::CollisionGeometry*>(geometries[i]));
  auto result=bp::import("numpy").attr("empty")(bp::make_tuple(count,10), "float64");
  Buffer output(result,true);
  auto pp=static_cast<const double*>(p.view.buf), rr=static_cast<const double*>(r.view.buf);
  auto ids=static_cast<const std::int64_t*>(c.view.buf);
  auto values=static_cast<double*>(output.view.buf);
  std::vector<coal::Transform3s> transforms(batch*shapes);
  for (Py_ssize_t i=0; i<batch*shapes; ++i) {
    coal::Matrix3s rotation; coal::Vec3s position;
    for (int x=0; x<3; ++x) {
      position[x]=pp[3*i+x];
      for (int y=0; y<3; ++y) rotation(x,y)=rr[9*i+3*x+y];
    }
    transforms[i]=coal::Transform3s(rotation,position);
  }
  for (Py_ssize_t k=0; k<count; ++k) {
    auto sample=ids[3*k], i=ids[3*k+1], j=ids[3*k+2];
    if (sample<0 || sample>=batch || i<0 || i>=shapes || j<0 || j>=shapes)
      throw std::invalid_argument("Invalid candidate index");
    coal::DistanceResult distance;
    values[10*k]=coal::distance(geometry[i], transforms[sample*shapes+i],
                               geometry[j], transforms[sample*shapes+j], request, distance);
    for (int x=0; x<3; ++x) {
      values[10*k+1+x]=distance.normal[x];
      values[10*k+4+x]=distance.nearest_points[0][x];
      values[10*k+7+x]=distance.nearest_points[1][x];
    }
  }
  return result;
}
BOOST_PYTHON_MODULE(coal_batch) { bp::def("run",run); }
