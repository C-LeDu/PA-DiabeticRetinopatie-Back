import cv2
import numpy as np
from flask import Flask
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


file_upload = reqparse.RequestParser()
file_upload.add_argument('png_file',
                         type=FileStorage,
                         location='files',
                         required=True,
                         help='PNG file')


@api.route('/signIn',  endpoint='with-parser')
class SignIn(Resource):
    @api.expect(data)
    def get(self):
        args = data.parse_args(strict=True)
        print(args['email'] + " " + args['psw'])
        return {'token': 'token'}


@api.route('/upload')
class my_file_upload(Resource):
    @api.expect(file_upload)
    def post(self):
        args = file_upload.parse_args()
        if args['png_file'].mimetype == 'image/png':
            img = cv2.imdecode(np.fromstring(args['png_file'].read(), np.uint8), cv2.IMREAD_COLOR)
        else:
            abort(400, 'error when get the png_file')
        return {'status': 'Done'}


if __name__ == '__main__':
    app.run(debug=True)
