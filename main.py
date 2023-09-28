from fastapi import FastAPI
from fastapi.responses import JSONResponse
from recSys.recommendationSystem import simple_movie_recommendation, aggregated_movie_recomendation, data
from traceback import print_exception
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={'message': 'Hello World'},
    )


@app.get("/movie_predict/{movie_name}")
async def simple_recommendation(movie_name: str):
    try:
        movies_list = simple_movie_recommendation(movie_name)

        return JSONResponse(
            status_code=200,
            content={'movies': movies_list},
        )

    except Exception as e:
        print_exception(e)
        return JSONResponse(
            content={'message': 'Internal server error'},
            status_code=500
        )


@app.get("/movie_predict_grouped/{movie_name}")
async def grouped_recommendation(movie_name: str):
    try:
        movies_list = aggregated_movie_recomendation(movie_name)

        return JSONResponse(
            status_code=200,
            content={'movies': movies_list},
        )

    except Exception as e:
        print_exception(e)
        return JSONResponse(
            content={'message': 'Internal server error'},
            status_code=500
        )


@app.get("/movies_list/")
async def movie_list():
    try:
        df = data()
        movies_list = sorted(list(df[df.vote_count >= 50].original_title.unique()))

        return JSONResponse(
            status_code=200,
            content={'movies': movies_list},
        )

    except Exception as e:
        print_exception(e)
        return JSONResponse(
            content={'message': 'Internal server error'},
            status_code=500
        )
