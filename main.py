import io
import os
import uuid

import keras
import numpy as np
from PIL import Image
from flask import Flask
from flask import send_file, make_response
from flask_cors import CORS
from flask_restplus import Resource, Api, reqparse
from flask_restplus import abort
from werkzeug.datastructures import FileStorage

import config
from PredictionPipeline import PredictionPipeline

app = Flask(__name__)
app.config['BUNDLE_ERRORS'] = True
api = Api(app, version="1.0", title="Back-end")
CORS(app)

keras.backend.clear_session()
PP = PredictionPipeline()

# data = reqparse.RequestParser()
# data.add_argument('email', type=str, required=True, location='args')
# data.add_argument('psw', type=str, required=True, location='args')
#
#
# @api.route('/signIn',  endpoint='with-parser')
# class SignIn(Resource):
#     @api.expect(data)
#     def get(self):
#         args = data.parse_args(strict=True)
#         print(args['email'] + " " + args['psw'])
#         return {'token': 'token'}

image_file_upload = reqparse.RequestParser()
image_file_upload.add_argument('image_file',
                               type=FileStorage,
                               location='files',
                               required=True,
                               help='image file')


def convert_values_to_file_name(values):
    str_values = np.array2string(values, formatter={'float_kind': lambda x: "%.2f" % x})\
        .replace("[[", "")\
        .replace("]]", "")\
        .replace(" ", "-")
    return str(np.argmax(values)) + "_" + str_values + ".jpeg"


@api.route('/predict')
class MyFileUpload(Resource):
    @api.expect(image_file_upload)
    def post(self):
        args = image_file_upload.parse_args()
        if args['image_file'].mimetype == 'image/png' or args['image_file'].mimetype == 'image/jpeg':
            img = Image.open(args['image_file'].stream).convert('RGB')
            img.thumbnail((512, 512), Image.ANTIALIAS)
            path = "tmp/" + str(uuid.uuid4())+".jpeg"
            img.save(path, "JPEG")
            # Make all we need with model
            values, img = PP.predict_image(path)
            file = io.BytesIO()
            img.save(file, 'jpeg')
            file.seek(0)
            # values = np.random.dirichlet(np.ones(2), size=1)
            response = make_response(send_file(file,
                                               as_attachment=True,
                                               attachment_filename=convert_values_to_file_name(values),
                                               mimetype='image/jpeg'))
            response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
            response.headers['Access-Control-Allow-Origin'] = '*'
            os.remove(path)
            return response
        else:
            abort(400, 'error when get the image file')
        return {'status': 'Done'}


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=config.PORT, debug=config.DEBUG_MODE)
