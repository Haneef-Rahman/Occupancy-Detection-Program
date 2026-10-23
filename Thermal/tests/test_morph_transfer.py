import unittest,json
from pathlib import Path
import numpy as np
from morph_field import MorphField,DEFAULT_MORPH
from rgb_annotation_worker import transfer_box,rgb_to_ir_matrix

class Shift:
 def __init__(self,d):self.d=np.array(d);self.calls=[]
 def displacement(self,c):
  self.calls.append(np.array(c));return self.d,dict(displacement_ir=self.d.tolist(),inside_support=True,extrapolated=False)

class MorphTests(unittest.TestCase):
 def test_uniform_translation_and_actual_box_centre(self):
  shift=Shift([2,-3]);audit={};box=transfer_box([10,20,20,40],np.eye(3),morph=shift,audit=audit)
  np.testing.assert_allclose(box,[12,17,22,37]);np.testing.assert_allclose(shift.calls,[[15,30]])
  self.assertFalse(audit['clipped'])
 def test_shift_before_any_clipping_can_recover_an_edge_box(self):
  box=transfer_box([-8,10,-2,20],np.eye(3),morph=Shift([10,0]))
  np.testing.assert_allclose(box,[2,10,8,20])
 def test_clipping_after_shift_and_outside_rejection(self):
  audit={};box=transfer_box([150,20,158,30],np.eye(3),morph=Shift([5,0]),audit=audit)
  np.testing.assert_allclose(box,[155,20,160,30]);self.assertTrue(audit['clipped'])
  self.assertIsNone(transfer_box([150,20,158,30],np.eye(3),morph=Shift([20,0])))
 def test_rotation_then_translation_uses_projected_centre(self):
  m=rgb_to_ir_matrix({'state':dict(x=800,y=600,scale=10,angle=90)})
  shift=Shift([1,2]);box=transfer_box([700,400,900,800],m,morph=shift)
  np.testing.assert_allclose(box,[61,52,101,72]);np.testing.assert_allclose(shift.calls,[[80,60]])
 def test_saved_field_inside_and_extrapolated(self):
  meta=json.loads(DEFAULT_MORPH.read_text());field=MorphField(DEFAULT_MORPH,meta['alignment_sha256'])
  a={};transfer_box([75,55,85,65],np.eye(3),morph=field,audit=a);self.assertTrue(a['inside_support'])
  b={};transfer_box([1,1,7,7],np.eye(3),morph=field,audit=b);self.assertTrue(b['extrapolated'])
  self.assertEqual(field.provenance()['fit_points'],69)
 def test_wrong_alignment_fails_before_use(self):
  with self.assertRaisesRegex(ValueError,'alignment'):MorphField(DEFAULT_MORPH,'wrong')
 def test_invalid_source_never_evaluates_field(self):
  shift=Shift([1,1]);self.assertIsNone(transfer_box([0,0,np.nan,1],np.eye(3),morph=shift));self.assertEqual(shift.calls,[])
if __name__=='__main__':unittest.main()
