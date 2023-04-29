import trimesh 


#mesh = trimesh.creation.box(extents=(4, 4, 1))
mesh = trimesh.exchange.load.load_mesh('0000/scene_box_8.obj')
trimesh.exchange.urdf.export_urdf(mesh, 'ase/preprocess/scene_box_8', scale=1, color=None )

#mesh.show()