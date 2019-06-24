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
from flask import Flask, render_template, request

app = Flask(__name__)
app.config['BUNDLE_ERRORS'] = True
api = Api(app, version="1.0", title="Back-end")
CORS(app)

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


@api.route('/predict')
class MyFileUpload(Resource):
    @api.expect(image_file_upload)
    def post(self):
        args = image_file_upload.parse_args()
        if args['image_file'].mimetype == 'image/png' or args['image_file'].mimetype == 'image/jpeg':
            img = cv2.cvtColor(cv2.imdecode(np.fromstring(args['image_file'].read(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            # Make all we need with model
            file = io.BytesIO()
            Image.fromarray(img, 'RGB').save(file, 'jpeg')
            file.seek(0)
            return send_file(file,
                             as_attachment=True,
                             attachment_filename='annotate.jpeg',
                             mimetype='image/jpeg')
        else:
            abort(400, 'error when get the image file')
        return {'status': 'Done'}


@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/submitted', methods=['POST'])
def submitted_form():
    name = request.form['name']
    email = request.form['email']
    site = request.form['site_url']
    comments = request.form['comments']

    return render_template(
        'submitted_form.html',
        name=name,
        email=email,
        site=site,
        comments=comments)

if __name__ == '__main__':
    app.run(debug=True)
