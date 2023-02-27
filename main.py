from flask import Flask, render_template, request, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, HiddenField, DecimalField, DateField
from wtforms.validators import DataRequired
from flask_sqlalchemy import SQLAlchemy
import settings
import utils
import numpy as np
import cv2

# db = sqlite3.connect("books-collection.db")
#
# cursor = db.cursor()
# #cursor.execute("CREATE TABLE books (id INTEGER PRIMARY KEY, title varchar(250) NOT NULL UNIQUE, author varchar(250) NOT NULL, rating FLOAT NOT NULL)")
# cursor.execute("INSERT OR IGNORE INTO books VALUES(1, 'Harry Potter', 'J. K. Rowling', '9.3')")
# db.commit()

app = Flask(__name__)
# Push the app in the app context - otherwise it will produce an error
app.app_context().push()
app.config['SECRET_KEY'] = '8BYkEfBA6O6donzWlSihBXox7C0sKR6b'
Bootstrap(app)
##CREATE DATABASE
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///timesheet-collection.db'
#Optional: But it will silence the deprecation warning in the console.
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

docscan = utils.DocumentScan()

##CREATE TABLE
class TS(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(250), unique=False, nullable=False)
    first_name = db.Column(db.String(250), unique=False, nullable=False)
    project = db.Column(db.String(250), unique=False, nullable=False)
    bestnr = db.Column(db.String(250), unique=False, nullable=True)
    posnr = db.Column(db.Integer, unique=False, nullable=True)
    pt = db.Column(db.Float, unique=False, nullable=False)
    pt_onsite = db.Column(db.Float, unique=False, nullable=True)
    date = db.Column(db.String(250), unique=False, nullable=False)

    # Optional: this will allow each timesheet object to be identified by its title when printed.
    # def __repr__(self):
    #     return f'<Timesheet {self.title}>'


db.create_all()

#Create A New Record
# new_entry = Book(id=1, title='Harry Potter', author='Joanne K. Rowling', rating=9.3)
# db.session.add(new_entry)
# db.session.commit()

# all_books = []
# all_books = db.session.query(Book).all()
# print(all_books)


class TSForm(FlaskForm):
    id = HiddenField("id")
    name = StringField('Last Name', validators=[DataRequired()])
    first_name = StringField("First Name", validators=[DataRequired()])
    project = StringField("Project", validators=[DataRequired()])
    bestnr = StringField("Order number")
    posnr = DecimalField("Position number", places=0)
    pt = DecimalField("Total mandays", places=4, validators=[DataRequired()])
    pt_onsite = DecimalField("Mandays onsite", places=4)
    date = DateField("Timesheet Date - last day of the month", format='%d-%m-%Y', validators=[DataRequired()])
    submit = SubmitField('Submit')


# Exercise:
# add: Location URL, open time, closing time, coffee rating, wifi rating, power outlet rating fields
# make coffee/wifi/power a select element with choice of 0 to 5.
#e.g. You could use emojis ☕️/💪/✘/🔌
# make all fields required except submit
# use a validator to check that the URL field has a URL entered.
# ---------------------------------------------------------------------------


#Create A New Record
# new_book = Book(id=1, title="Harry Potter", author="J. K. Rowling", rating=9.3)
# db.session.add(new_book)
# db.session.commit()

# #NOTE: When creating new records, the primary key fields is optional. you can also write:
#
# new_book = Book(title="Harry Potter", author="J. K. Rowling", rating=9.3)
#
# #the id field will be auto-generated.
#
#
# #Read All Records
#
# all_books = db.session.query(Book).all()
#
#
# #Read A Particular Record By Query
#
# book = Book.query.filter_by(title="Harry Potter").first()
#
#
# #Update A Particular Record By Query
#
# book_to_update = Book.query.filter_by(title="Harry Potter").first()
# book_to_update.title = "Harry Potter and the Chamber of Secrets"
# db.session.commit()
#
#
# #Update A Record By PRIMARY KEY
#
# book_id = 1
# book_to_update = Book.query.get(book_id)
# book_to_update.title = "Harry Potter and the Goblet of Fire"
# db.session.commit()
#
#
# #Delete A Particular Record By PRIMARY KEY
#
# book_id = 1
# book_to_delete = Book.query.get(book_id)
# db.session.delete(book_to_delete)
# db.session.commit()

# all Flask routes below
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["image_name"]
        upload_image_path = utils.save_upload_image(file)
        print("Image saved in this location ", upload_image_path)
        # predict the coordination of the document
        four_points, size = docscan.document_scanner(upload_image_path)
        if four_points is None:
            message = "Unable to locate coordinates of document: Coordinates displayed are random"
            points = [
                {"x": 10, "y": 10},
                {"x": 120, "y": 10},
                {"x": 120, "y": 120},
                {"x": 10, "y": 120},
            ]
            return render_template("scanner.html", points=points, fileupload=True, message=message)
        else:
            points = utils.array_to_json(four_points)
            message = "Edges of uploaded file located - please check and adjust if necessary"
            return render_template("scanner.html", points=points, fileupload=True, message=message)

        # return render_template("index.html")
    return render_template("index.html")


@app.route('/transform', methods=['POST'])
def transform():
    try:
        points = request.json['data']
        array = np.array(points)
        magic_color = docscan.calibrate_to_original_size(array)
        # utils.save_image(magic_color,'magic_color.jpg')
        filename = 'magic_color.jpg'
        magic_image_path = settings.join_path(settings.MEDIA_DIR, filename)
        cv2.imwrite(magic_image_path, magic_color)

        return 'success'
    except:
        return 'fail'

# @app.route('/prediction')
# def prediction():
#     return "success"


@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        new_ts = TS(name=request.form.get("last_name"),
                    first_name=request.form.get("first_name"),
                    project=request.form.get("project"),
                    bestnr=request.form.get("bestnr"),
                    posnr=request.form.get("posnr"),
                    pt=request.form.get("pt"),
                    pt_onsite=request.form.get("pt_onsite"),
                    date=request.form.get("date")
                    )
        db.session.add(new_ts)
        db.session.commit()
        return redirect(url_for('home'))
    # load the wrap image
    # form = TSForm()
    wrap_image_filepath = settings.join_path(settings.MEDIA_DIR, 'magic_color.jpg')
    # image = cv2.imread(wrap_image_filepath)
    results = utils.new_final_prediction(wrap_image_filepath)
    # if form.validate_on_submit():

    # bb_filename = settings.join_path(settings.MEDIA_DIR, 'bounding_box.jpg')
    # cv2.imwrite(bb_filename, image_bb)

    return render_template('predictions.html', results=results)


# @app.route("/add", methods=["POST"])
# def post_new_ts():
#     new_ts = TS(name=request.form.get("last_name"),
#                 first_name=request.form.get("first_name"),
#                 project=request.form.get("project"),
#                 bestnr=request.form.get("bestnr"),
#                 posnr=request.form.get("posnr"),
#                 pt=request.form.get("pt"),
#                 pt_onsite=request.form.get("pt_onsite"),
#                 date=request.form.get("date")
#                 )
#     db.session.add(new_ts)
#     db.session.commit()
#     return redirect(url_for('home'))

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         # four_points = None
#         file = request.files['image_name']
#         upload_image_path = utils.save_upload_image(file)
#         print('Image saved in = ', upload_image_path)
#         # predict the coordination of the document
#         four_points, size = docscan.document_scanner(upload_image_path)
#         print(four_points, size)
#         if four_points is None:
#             message = 'UNABLE TO LOCATE THE COORDIANATES OF DOCUMENT: points displayed are random'
#             points = [
#                 {'x': 10, 'y': 10},
#                 {'x': 120, 'y': 10},
#                 {'x': 120, 'y': 120},
#                 {'x': 10, 'y': 120}
#             ]
#             return render_template('scanner.html',
#                                    points=points,
#                                    fileupload=True,
#                                    message=message)
#
#         else:
#             points = utils.array_to_json(four_points)
#             message = 'Located the Cooridinates of Document using OpenCV'
#             return render_template('scanner.html',
#                                    points=points,
#                                    fileupload=True,
#                                    message=message)
#
#         # return render_template('index.html')
#
#     return render_template('scanner.html')



@app.route("/about")
def about():
    return render_template("about.html")


# @app.route('/add', methods=["GET", "POST"])
# def add_book():
#     form = BookForm()
#     if form.validate_on_submit():
#         new_book = Book(title=form.title.data, author=form.author.data, rating=form.rating.data)
#         db.session.add(new_book)
#         db.session.commit()
#
#         #+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#         # print("True")
#         # temp_dict = {
#         #     "title": form.title.data,
#         #     "author": form.author.data,
#         #     "rating": form.rating.data
#         # }
#         # all_books.append(temp_dict)
#         # print(all_books)
#     #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#         # with open("cafe-data.csv", mode="a", encoding="utf-8") as csv_file:
#         #     csv_file.write(f"\n{form.cafe.data},"
#         #                    f"{form.location.data},"
#         #                    f"{form.open.data},"
#         #                    f"{form.close.data},"
#         #                    f"{form.coffee_rating.data},"
#         #                    f"{form.wifi_rating.data},"
#         #                    f"{form.power_rating.data}")
#
#         return redirect(url_for('home'))
#     # Exercise:
#     # Make the form write a new row into cafe-data.csv
#     # with   if form.validate_on_submit()
#     return render_template('add.html', form=form)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
