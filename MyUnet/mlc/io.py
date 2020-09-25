# -*- coding: utf-8 -*-
import os
import re
import vtk
import gzip
import numpy as np
from scipy import ndimage as ndi
from vtk.util import numpy_support


def read_raw(filename):
    """
    mhdデータの読み込み．

    Parameters
    ----------
    filename : string
        mhdデータのパスを指定．

    Returns
    -------
    image : vtk型のimage画像．
    """
    image = vtk.vtkMetaImageReader()
    image.SetFileName(filename)
    image.Update()

    return image

#https://docs.python.jp/3.3/library/re.html
def mhd_read_header_file(filename):
    with open(filename + ".header", 'r') as fid:
        print(filename)
        contents = fid.read()

    info = dict()
    for line in contents.split('\n'):
        m = re.match('\A(\w+)', line)

       # print(m)

        if m is None:
            continue

        key = m.groups()[0].lower()
       #print(key)
        m = re.match('\A\w+ *= * (.*) *', line)

        print(line)

        if m is None:
            continue

        data = m.groups()[0]
        
        if re.match('.*[^0-9 \.].*', data) != None:
          info[key] = data
          #print("\n\n\n==========================================")
          continue

        data = data.split(' ')
        numbers = []
        for number in data:
            if len(number) > 0:
                numbers.append(float(number))

        if len(numbers) == 1:
            numbers = float(numbers[0])

        info[key] = numbers
        print(info[key])
    
    return info


def read_raw_gz(filename):
    """
    mhdデータの読み込み．

    Parameters
    ----------
    filename : string
        mhdデータのパスを指定．

    Returns
    -------
    image : vtk型のimage画像．
    """
    root, ext = os.path.splitext(filename)
    data_name = root + ".raw.gz"

    info = mhd_read_header_file(filename)
    spacing = info['elementspacing']
    x, y, z = info['dimsize']
    type = info['elementtype']
    if type == 'MET_SHORT':
        data_type = np.int16
    elif type == 'MET_UCHAR':
        data_type = np.uint8

    with gzip.open(data_name, 'r') as f:
        binary = f.read()
        image = np.fromstring(binary, dtype=data_type)
        image = np.reshape(image, (int(x), int(y), int(z)), order='F')
        image = numpy_to_vtk(image, tuple(spacing))
    
    return image

def O_read_raw_gz(filename,a,b,c):
    """
    mhdデータの読み込み．

    Parameters
    ----------
    filename : string
        mhdデータのパスを指定．

    Returns
    -------
    image : vtk型のimage画像．
    """
    root, ext = os.path.splitext(filename)
    data_name = root + ".gz"
    
    info = mhd_read_header_file(filename)
    spacing = (info['pitchx'],info['pitchy'],info['pitchz'])
    #spacing = (0,0,1)
    x = a
    y = b
    z = c
    type = 'MET_SHORT'
    if type == 'MET_SHORT':
        data_type = np.int16
    elif type == 'MET_UCHAR':
        data_type = np.uint8
    print(data_name)
    with gzip.open(data_name, 'r') as f:
        binary = f.read()
        image = np.fromstring(binary, dtype=data_type)
        image = np.reshape(image, (int(x), int(y), int(z)), order='F')
        image = numpy_to_vtk(image, tuple(spacing))
    
    return image

def read_dicom(filename):
    """
    dicomデータの読み込み．

    Parameters
    ----------
    filename : string
        dicomデータのディレクトリへのパスを指定．

    Returns
    -------
    image : vtk型のimage画像．
    """
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(filename)
    reader.Update()

    # vtkとDICOM画像において，空間座標系が違うため変換が必要．
    # 詳細は，“https://itk.org/Wiki/Proposals:Orientation”
    flip = vtk.vtkImageFlip()
    flip.SetFilteredAxis(1)
    flip.SetInputData(reader.GetOutput())
    flip.SetOutputSpacing(reader.GetDataSpacing())
    flip.Update()
    image = flip

    return image


def read_nifti(filename):
    """
    NIfTI-1（.nii）の単一ファイル形式の読み込み．

    Parameters
    ----------
    filename : string
        .niiまたは.nii.gz形式のパスを指定．

    Returns
    -------
    image : vtk型のimage画像．
    """
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(filename)
    reader.Update()

    flip = vtk.vtkImageFlip()
    flip.SetFilteredAxis(2)
    flip.SetInputData(reader.GetOutput())
    flip.Update()
    image = flip

    return image


def create_header(image, filename):
    """
    PLUTOで読み込めるヘッダーファイルの作成．

    Parameters
    ----------
    image : vtk data
    filename : string
    """
    path, data_name = os.path.split(filename)
    header_name = data_name + ".header"
    save_file = os.path.join(path, header_name).replace("\\", "/")

    w, h, d = image.GetOutput().GetDimensions()
    x, y, z = image.GetOutput().GetSpacing()
    
    f = open(save_file, 'w')
    f.write('OrgFile\t : {}\r'.format(data_name))
    f.write('SizeX\t : {}\r'.format(w))
    f.write('SizeY\t : {}\r'.format(h))
    f.write('SizeZ\t : {}\r'.format(d))
    f.write('PitchX\t : {0:.3f}\r'.format(x))
    f.write('PitchY\t : {0:.3f}\r'.format(y))
    f.write('PitchZ\t : {0:.3f}\r'.format(z))
    f.close()


def write_raw(image, filename, compression=False):
    """
    rawデータの書き込み．

    Parameters
    ----------
    image : vtk data
    filename : string
        mhdまたはmha形式のパスを指定．
    """
    root, ext = os.path.splitext(filename)
    raw_name = root + ".raw"
    create_header(image, raw_name)

    writer = vtk.vtkMetaImageWriter()
    writer.SetInputData(image.GetOutput())
    writer.SetFileName(filename)
    writer.SetCompression(compression)
    
    writer.Write()


def write_raw_gz(image, filename):
    """
    raw.gzデータの書き込み．

    Parameters
    ----------
    image : vtk data
    filename : string
        mhdまたはmha形式のパスを指定．
    """
    root, ext = os.path.splitext(filename)
    raw_name = root + ".raw.gz"
    create_header(image, raw_name)

    writer = vtk.vtkMetaImageWriter()
    writer.SetInputData(image.GetOutput())
    writer.SetFileName(filename)
    writer.SetRAWFileName(raw_name)
    writer.SetCompression(False)
    writer.Write()

    with open(raw_name, 'rb') as f:
        xml = f.read()

    with gzip.open(raw_name, 'w') as gzfile:
        gzfile.write(xml)


def shiftscale_uchar(image, ww=300.0, wc=40.0):
    """
    Parameters
    ----------
    image : vtk data
    ww : float
        window width
    wc : float
        window center

    Returns
    -------
    shift_data : vkt uchar data
    """
    shift_scale = vtk.vtkImageShiftScale()
    shift_scale.SetInputData(image.GetOutput())

    shift_scale.SetShift(-(wc - 0.5 * ww))


    shift_scale.SetScale(255.0 / ww)

    shift_scale.SetOutputScalarTypeToUnsignedChar()
    shift_scale.ClampOverflowOn()
    shift_scale.Update()

    return shift_scale


def shiftscale_float(image):
    """
    Parameters
    ----------
    image : vtk data

    Returns
    -------
    shift_data : vkt float data
    """

    shift_scale = vtk.vtkImageShiftScale()
    shift_scale.SetInputData(image.GetOutput())

    shift_scale.SetOutputScalarTypeToFloat()
    shift_scale.ClampOverflowOn()
    shift_scale.Update()

    return shift_scale


def vtk_to_numpy(image):
    """
    Parameters
    ----------
    image : vtk data

    Returns
    -------
    numpy data : numpy array 
    """
    dim = image.GetOutput()
    width, height, depth = dim.GetDimensions()
    sc = dim.GetPointData().GetScalars()
    output = numpy_support.vtk_to_numpy(sc)
    result = np.reshape(output, (width, height, depth), order='F')

    return result


def numpy_to_vtk(numpy_data, spacing):
    """
    Parameters
    ----------
    image : 3d numpy array
    spacing : 解像度情報．(float, float, float)のタプル型．

    Returns
    -------
    numpy data : vtk data 
    """
    w, h, d = numpy_data.shape
    scalar_type = numpy_support.get_vtk_array_type(numpy_data.dtype)

    # データ順序がC形式でなければならないため，Fortran形式から変換．
    if numpy_data.flags.c_contiguous != True:
        numpy_data = np.ndarray.flatten(numpy_data, order='F')
        numpy_data = np.reshape(numpy_data, (w, h, d), order='C')

    vtk_data = vtk.vtkImageImport()
    vtk_data.SetDataScalarType(scalar_type)
    vtk_data.SetDataSpacing(spacing)
    vtk_data.SetDataOrigin(0, 0, 0)
    vtk_data.SetDataExtent(0, w - 1, 0, h - 1, 0, d - 1)
    vtk_data.SetWholeExtent(0, w - 1, 0, h - 1, 0, d - 1)
    vtk_data.SetNumberOfScalarComponents(1)
    vtk_data.CopyImportVoidPointer(numpy_data, numpy_data.nbytes)
    vtk_data.Update()

    return vtk_data

def get_ext_lists(path, ext):
    """
    指定した拡張子のパスの検索．指定フォルダ以下のディレクトリを再帰的に検索．
    dicom形式は，フォルダへのパスである必要があるため未対応．get_dicom_dirsの利用を推奨．

    Parameters
    ----------
    path : string
        検索するフォルダのパス．
    ext : string
        検索する拡張子の指定．

    Returns
    -------
    file_lists : 指定した拡張子のパスのリスト．
    """
    file_lists = []
    for (root, dirs, files) in os.walk(path):
        for file in files:
            target = os.path.join(root, file).replace("\\", "/")
            if os.path.isfile(target):
                if re.search(ext, target) != None:
                    file_lists.append(target)

    return file_lists


def voxel_to_cube(image, spacing):
    """

    """
    reso1, reso2, reso3 = spacing
    w, h, d = image.shape
    r_w = w * (1.0 / reso1)
    r_h = h * (1.0 / reso2)
    r_d = d * (1.0 / reso3)

    output = np.zeros((int(round(r_w)), int(round(r_h)), int(round(r_d))), dtype=image.dtype, order='F')
    ndi.interpolation.zoom(image, ((1.0 / reso1), (1.0 / reso2), (1.0 / reso3)), output=output)

    return output

def cube_to_voxel(image, spacing):
    """

    """
    reso1, reso2, reso3 = spacing
    w, h, d = image.shape
    r_w = w * reso1
    r_h = h * reso2
    r_d = d * reso3

    output = np.zeros((int(round(r_w)), int(round(r_h)), int(round(r_d))), dtype=image.dtype, order='F')
    ndi.interpolation.zoom(image, (reso1, reso2, reso3), output=output)

    return output