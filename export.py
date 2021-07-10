from dolfin import Mesh, Expression, Function, Facet, Edge, vertices, cells, edges
from vtk import vtkUnstructuredGrid, vtkPolyData, vtkXMLUnstructuredGridWriter, vtkXMLUnstructuredGridReader, vtkXMLPolyDataWriter, vtkXMLPolyDataReader
from vtk import vtkPoints, vtkTetra, vtkVertex, VTK_TETRA, vtkCellArray, vtkDoubleArray
import math
import os.path
import numpy as np
import numpy.linalg
import scipy.integrate
import scipy.sparse
import scipy.io

# FIXME: works only for linear elements
class DOFWriter(object):
    """
    dofs = DOFWriter(mesh, E.vector(), V)
    dofs.start()
    points, values = dofs.stop()
    d = DataFile(points)
    d.add_data(values, name="Values")
    d.write("results/DOF.vtp")
    """
    def __init__(self, mesh, solution, function_space):
        self.mesh = mesh
        self.solution = solution
        self.function_space = function_space
        n = mesh.num_edges()
        m = function_space.num_sub_spaces()
        self.values = np.zeros((n, m))
        self.points = np.zeros((n, 3))

    def start(self):
        n, m = self.values.shape
        dofs = []
        for j in range(m):
            dofs.append(self.function_space.sub(j).dofmap().dofs(self.mesh, 1))
        for edge in edges(self.mesh):
            i = edge.global_index()
            p = edge.midpoint()
            for j in range(m):
                dof = dofs[j][i]
                self.values[i,j] = self.solution[dof]
            self.points[i,:] = [p.x(), p.y(), p.z()]

    def stop(self):
        return self.points, self.values

# FIXME: works only for linear elements
def extract(mesh, boundaries, boundary, values):
    mesh.init(1, 0) # edges -> vertices
    mesh.init(2, 0) # facets -> vertices
    mesh.init(2, 1) # facets -> edges

    element_indices = (boundaries.array() == boundary).nonzero()[0]
    n = len(element_indices)
    element_vertex_coordinates = np.zeros((n, 3*3))
    element_edge_values = np.zeros((n, 3))

    for i in element_indices:
        n -= 1
        f = Facet(mesh, i)
        v = f.entities(0)
        e = f.entities(1)
        element_vertex_coordinates[n,:] = mesh.coordinates()[v].ravel()
        for j in e:
            ed = Edge(mesh, j)
            ind = np.in1d(v, ed.entities(0))
            if ind[0] and ind[1] and not ind[2]:
                element_edge_values[n,0] = +values[j]
            if ind[0] and not ind[1] and ind[2]:
                element_edge_values[n,1] = -values[j]
            if not ind[0] and ind[1] and ind[2]:
                element_edge_values[n,2] = +values[j]
    return element_vertex_coordinates, element_edge_values

class DOFRecorder(Expression):
    """
    eTEM = DOFRecorder(Eport)
    eTEM.start()
    assemble(inner(eTEM, T_re) * dSin)
    points, values = eTEM.stop()
    d = DataFile(points)
    d.add_data(values, name="Values")
    d.write("results/DOF.vtp")
    """
    def __init__(self, expr):
        self.expr = expr
        self.values = None
        self.points = None
        Expression.__init__(self)

    def eval_cell(self, values, x, ufc_cell):
        self.expr.eval_cell(values, x, ufc_cell)
        if self.values is not None and self.points is not None:
            self.points.append([x[0], x[1], x[2]])
            self.values.append([values[0], values[1], values[2]])

    def eval(self, values, x):
        self.expr.eval(values, x)

    def value_shape(self):
        return self.expr.value_shape()

    def start(self):
        self.values = []
        self.points = []

    def stop(self):
        points = np.array(self.points)
        values = np.array(self.values)
        self.points = None
        self.value = None
        return points, values

class ResultFile(object):
    def __init__(self, mesh):
        self.nv, self.nc = mesh.num_vertices(), mesh.num_cells()
        self.grid = self.convert_mesh_to_grid(mesh)

    def convert_mesh_to_grid(self, mesh):
        grid = vtkUnstructuredGrid()
        points = vtkPoints()
        cell_array = vtkCellArray()
        for v in vertices(mesh):
            vp = v.point()
            points.InsertNextPoint(vp.x(), vp.y(), vp.z())
        for c in cells(mesh):
            t = vtkTetra()
            for i, v in enumerate(c.entities(0)):
                t.GetPointIds().SetId(i, v)
            cell_array.InsertNextCell(t)
        grid.SetPoints(points)
        grid.SetCells(VTK_TETRA, cell_array)
        return grid

    def get_data_from_array(self, A):
        if A.size == len(A):
            entries, components = len(A), 1
        else:   
            entries, components = A.shape
        data = vtkDoubleArray()
        data.SetNumberOfComponents(components)
        data.SetNumberOfTuples(entries)
        for i, entry in enumerate(A):
            if components == 1:
                data.SetTuple1(i, entry)
            if components == 2:
                data.SetTuple2(i, entry[0], entry[1])
            if components == 3:
                data.SetTuple3(i, entry[0], entry[1], entry[2])
        return data

    def get_data_from_function(self, f):
        entries, components = self.nv, f.value_shape()[0]
        A = f.compute_vertex_values()
        A = A.reshape((entries, components), order='F')
        return self.get_data_from_array(A)

    def add_data(self, f, name=None):
        if name is None:
            name = f.name

        if isinstance(f, Function):
            data = self.get_data_from_function(f)
        else:
            data = self.get_data_from_array(f)
            
        data.SetName(name)
        n = data.GetNumberOfTuples()

        if n == self.nv:
            self.grid.GetPointData().AddArray(data)
        elif n == self.nc:
            self.grid.GetCellData().AddArray(data)
        else:
            self.grid.GetFieldData().AddArray(data)

    def write(self, file_name):
        writer = vtkXMLUnstructuredGridWriter()
        writer.SetFileName(file_name)
        writer.SetCompressorTypeToNone()
        writer.SetDataModeToAscii()
        writer.SetInputData(self.grid)
        writer.Write()

    def read(self, file_name):
        reader = vtkXMLUnstructuredGridReader()
        reader.SetFileName(file_name)
        reader.Update()
        if os.path.exists(file_name):
            self.grid = reader.GetOutput()
        else:
            print(file_name + " does not exist")

class DataFile(object):
    def __init__(self, points):
        self.nv = len(points)
        self.polydata = self.convert_points_to_polydata(points)
        self.points = points

    def convert_points_to_polydata(self, points):
        polydata = vtkPolyData()
        points_array = vtkPoints()
        vertex_array = vtkCellArray()
        for p in points:
            points_array.InsertNextPoint(p[0], p[1], p[2])
        for i in range(self.nv):
            v = vtkVertex()
            v.GetPointIds().SetId(0, i)
            vertex_array.InsertNextCell(v)
        polydata.SetPoints(points_array)
        polydata.SetVerts(vertex_array)
        return polydata

    def get_data_from_array(self, A):
        if A.size == len(A):
            entries, components = len(A), 1
        else:
            entries, components = A.shape
        data = vtkDoubleArray()
        data.SetNumberOfComponents(components)
        data.SetNumberOfTuples(entries)
        for i, entry in enumerate(A):
            if components == 1:
                data.SetTuple1(i, entry)
            if components == 2:
                data.SetTuple2(i, entry[0], entry[1])
            if components == 3:
                data.SetTuple3(i, entry[0], entry[1], entry[2])
        return data

    def get_data_from_function(self, f):
        entries, components = self.nv, f.value_shape()
        A = evaluate_over_points(f, self.points)
        A = A.reshape((entries, components), order='F')
        return self.get_data_from_array(A)

    def add_data(self, f, name=None):
        if name is None:
            name = f.name

        if isinstance(f, Function):
            data = self.get_data_from_function(f)
        else:
            data = self.get_data_from_array(f)

        data.SetName(name)
        n = data.GetNumberOfTuples()

        if n == self.nv:
            self.polydata.GetPointData().AddArray(data)
        elif n == self.nc:
            self.polydata.GetCellData().AddArray(data)
        else:
            self.polydata.GetFieldData().AddArray(data)

    def write(self, file_name):
        writer = vtkXMLPolyDataWriter()
        writer.SetFileName(file_name)
        writer.SetCompressorTypeToNone()
        writer.SetDataModeToAscii()
        writer.SetInputData(self.polydata)
        writer.Write()

    def read(self, file_name):
        reader = vtkXMLPolyDataReader()
        reader.SetFileName(file_name)
        reader.Update()
        if os.path.exists(file_name):
            self.polydata = reader.GetOutput()
        else:
            print(file_name + " does not exist")

def save_matrices(filename, **kwargs):
    scipy.io.savemat(filename, kwargs)

def save_as_sparse_matrix(**kwargs):
    for arg in kwargs:
        dense_matrix = kwargs[arg]
        print("Saving %s.mat in sparse format" % arg)
        dofs = dense_matrix.size(0)
        print("DOFS = %d" % dofs)
        matrix = scipy.sparse.lil_matrix((dofs, dofs))
        for i in range(dofs):
            row = dense_matrix.getrow(i)
            matrix[i,row[0]] = row[1]
        scipy.io.savemat('%s.mat' % arg, {arg:matrix})
        print("...done")
