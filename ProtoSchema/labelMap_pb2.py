# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: labelMap.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='labelMap.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0elabelMap.proto\">\n\x0eLabelmapbuffer\x12\x0e\n\x06height\x18\x03 \x01(\r\x12\r\n\x05width\x18\x02 \x01(\r\x12\r\n\x05pixel\x18\x01 \x03(\r\"&\n\x16GeneratedImageResponse\x12\x0c\n\x04\x64\x61ta\x18\x04 \x03(\rb\x06proto3'
)




_LABELMAPBUFFER = _descriptor.Descriptor(
  name='Labelmapbuffer',
  full_name='Labelmapbuffer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='height', full_name='Labelmapbuffer.height', index=0,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='width', full_name='Labelmapbuffer.width', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='pixel', full_name='Labelmapbuffer.pixel', index=2,
      number=1, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=18,
  serialized_end=80,
)


_GENERATEDIMAGERESPONSE = _descriptor.Descriptor(
  name='GeneratedImageResponse',
  full_name='GeneratedImageResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='GeneratedImageResponse.data', index=0,
      number=4, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=82,
  serialized_end=120,
)

DESCRIPTOR.message_types_by_name['Labelmapbuffer'] = _LABELMAPBUFFER
DESCRIPTOR.message_types_by_name['GeneratedImageResponse'] = _GENERATEDIMAGERESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Labelmapbuffer = _reflection.GeneratedProtocolMessageType('Labelmapbuffer', (_message.Message,), {
  'DESCRIPTOR' : _LABELMAPBUFFER,
  '__module__' : 'labelMap_pb2'
  # @@protoc_insertion_point(class_scope:Labelmapbuffer)
  })
_sym_db.RegisterMessage(Labelmapbuffer)

GeneratedImageResponse = _reflection.GeneratedProtocolMessageType('GeneratedImageResponse', (_message.Message,), {
  'DESCRIPTOR' : _GENERATEDIMAGERESPONSE,
  '__module__' : 'labelMap_pb2'
  # @@protoc_insertion_point(class_scope:GeneratedImageResponse)
  })
_sym_db.RegisterMessage(GeneratedImageResponse)


# @@protoc_insertion_point(module_scope)
