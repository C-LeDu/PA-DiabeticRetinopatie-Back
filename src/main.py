import io

import cv2
import numpy as np
from PIL import Image
from flask import Flask
from flask import send_file
from flask_cors import CORS
from flask_restplus import Resource, Api, reqparse
from flask_restplus import abort
from werkzeug.datastructures import FileStorage

app = Flask(__name__)
app.config['BUNDLE_ERRORS'] = True
api = Api(app, version="1.0", title="Back-end")
CORS(app)


data = reqparse.RequestParser()
data.add_argument('email', type=str, required=True, location='args')
data.add_argument('psw', type=str, required=True, location='args')


image_file_upload = reqparse.RequestParser()
image_file_upload.add_argument('image_file',
                               type=FileStorage,
                               location='files',
                               required=True,
                               help='image file')


@api.route('/signIn',  endpoint='with-parser')
class SignIn(Resource):
    @api.expect(data)
    def get(self):
        args = data.parse_args(strict=True)
        print(args['email'] + " " + args['psw'])
        return {'token': 'token'}


@api.route('/upload')
class MyFileUpload(Resource):
    @api.expect(image_file_upload)
    def post(self):
        args = image_file_upload.parse_args()
        if args['image_file'].mimetype == 'image/png' or args['image_file'].mimetype == 'image/jpeg':
            ext = args['image_file'].mimetype.split('/')[1]
            img = cv2.imdecode(np.fromstring(args['image_file'].read(), np.uint8), cv2.IMREAD_COLOR)
            # Make all we need with model
            file = io.BytesIO()
            Image.fromarray(img).save(file, ext)
            file.seek(0)
            return send_file(file,
                             as_attachment=True,
                             attachment_filename='annotate.' + ext,
                             mimetype='image/' + ext)
        else:
            abort(400, 'error when get the image file')
        return {'status': 'Done'}


if __name__ == '__main__':
    app.run(debug=True)
