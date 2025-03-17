from flask import Flask, send_from_directory,request
from flask_restful import Api, Resource
from flask_cors import CORS
import xlsxwriter
import main


app = Flask(__name__)
CORS(app)
api = Api(app)
DOWNLOAD_DIRECTORY = "./"



@app.route("/get/<file_name>",methods = ['POST'])
def get(file_name):
    if request.method == 'POST':
        data = request.json
        print(data)
        ques_ans_distractors_arr = main.main(data[0],data[1])
        workbook = xlsxwriter.Workbook('mcqs.xlsx')
        worksheet = workbook.add_worksheet()
        cell_format = workbook.add_format()
        cell_format.set_text_wrap()
        worksheet.write('A1','Questions',cell_format)
        worksheet.write('B1','Answer',cell_format)
        worksheet.write('C1','Distractors/Options',cell_format)
        for row,arr in enumerate(ques_ans_distractors_arr):
            for col,data in enumerate(arr):
                if(col<2):
                    worksheet.write(row+1,col,data,cell_format)
                else:
                    worksheet.write(row+1,col,",".join(data),cell_format)
        workbook.close()
        return send_from_directory(DOWNLOAD_DIRECTORY,file_name, as_attachment=True)
    else:
        return send_from_directory(DOWNLOAD_DIRECTORY,file_name, as_attachment=True)

